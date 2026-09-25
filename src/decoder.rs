//! VTDecompressionSession-backed H.264 and HEVC decoders.
//!
//! # Design
//!
//! Each decoder holds a `VTDecompressionSessionRef` which is created
//! lazily the first time SPS+PPS (or VPS+SPS+PPS for HEVC) are seen.
//! Annex-B input is parsed to extract parameter sets, from which a
//! `CMVideoFormatDescription` is built. Subsequent NAL units are wrapped
//! in a `CMSampleBuffer` and handed to VideoToolbox.
//!
//! VT calls the registered callback with each decoded `CVPixelBuffer`
//! (NV12 / '420v'). Decoded buffers are retained as opaque hardware-frame
//! leases so consumers can import/sample them directly. The compatibility
//! `materialize()` path locks the buffer and copies/de-interleaves it to I420.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use oxideav_h264::access_unit::AnnexBAccessUnitAssembler;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, FrameLease, HardwareVideoFrame,
    HardwareVideoFrameStorage, Packet, PixelFormat, Result, VideoFrame, VideoPlane,
};

use crate::encoder::{build_hw_spec_dict, parse_hardware_mode, vt_error, HardwareMode};

use crate::sys::{
    self, CMSampleTimingInfo, CMTime, K_CV_PIXEL_FORMAT_420_YPCBCRi8_BI_PLANAR_VIDEO_RANGE,
    VTDecompressionOutputCallbackRecord, K_CV_PIXEL_BUFFER_LOCK_FLAGS_READ_ONLY,
    K_OS_STATUS_NO_ERROR,
};

// ─────────────────────────── NAL unit iterator ───────────────────────────────

fn annex_b_nals(buf: &[u8]) -> Vec<&[u8]> {
    let mut result = Vec::new();
    let mut pos = 0usize;
    let len = buf.len();

    while pos < len {
        // Find start code.
        let mut sc_len = 0usize;
        while pos + 3 <= len {
            if buf[pos] == 0 && buf[pos + 1] == 0 {
                if pos + 4 <= len && buf[pos + 2] == 0 && buf[pos + 3] == 1 {
                    sc_len = 4;
                    break;
                } else if buf[pos + 2] == 1 {
                    sc_len = 3;
                    break;
                }
            }
            pos += 1;
        }

        if sc_len == 0 {
            break;
        }
        pos += sc_len;
        let nal_start = pos;

        // Find end (next start code or end of buf).
        let mut nal_end = len;
        while pos + 3 <= len {
            if buf[pos] == 0 && buf[pos + 1] == 0 {
                if pos + 4 <= len && buf[pos + 2] == 0 && buf[pos + 3] == 1 {
                    nal_end = pos;
                    // Strip trailing zeros.
                    while nal_end > nal_start && buf[nal_end - 1] == 0 {
                        nal_end -= 1;
                    }
                    break;
                } else if buf[pos + 2] == 1 {
                    nal_end = pos;
                    while nal_end > nal_start && buf[nal_end - 1] == 0 {
                        nal_end -= 1;
                    }
                    break;
                }
            }
            pos += 1;
        }

        if nal_end > nal_start {
            result.push(&buf[nal_start..nal_end]);
        }
        // Don't advance pos past nal_end — the outer loop will rescan.
        pos = nal_end;
    }

    result
}

// ─────────────────────────── NAL type constants ───────────────────────────────

mod h264_nal {
    pub const SPS: u8 = 7;
    pub const PPS: u8 = 8;
}

mod hevc_nal {
    pub const VPS: u8 = 32;
    pub const SPS: u8 = 33;
    pub const PPS: u8 = 34;
}

// ─────────────────────────── Decoded buffer lease ─────────────────────────────

/// Retained VideoToolbox decoded frame backed by a CoreVideo CVPixelBuffer.
///
/// The pixel buffer remains valid until the last enclosing FrameLease is dropped.
/// Backend-aware consumers can import `pixel_buffer_ptr()` directly into CoreVideo /
/// Metal. Generic consumers can call `materialize()` for a CPU I420 copy.
pub struct VideoToolboxVideoFrameStorage {
    pixel_buffer: sys::CVPixelBufferRef,
    width: u32,
    height: u32,
    pts: Option<i64>,
}

unsafe impl Send for VideoToolboxVideoFrameStorage {}
unsafe impl Sync for VideoToolboxVideoFrameStorage {}

impl VideoToolboxVideoFrameStorage {
    pub fn pixel_buffer_ptr(&self) -> *mut c_void {
        self.pixel_buffer
    }
}

impl Drop for VideoToolboxVideoFrameStorage {
    fn drop(&mut self) {
        if !self.pixel_buffer.is_null() {
            if let Ok(vt) = sys::vtable() {
                unsafe { (vt.cf_release)(self.pixel_buffer) };
            }
        }
    }
}

impl HardwareVideoFrameStorage for VideoToolboxVideoFrameStorage {
    fn backend(&self) -> &'static str {
        "videotoolbox"
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::Nv12
    }

    fn pts(&self) -> Option<i64> {
        self.pts
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn materialize(&self) -> Result<VideoFrame> {
        materialize_pixel_buffer(self.pixel_buffer, self.width, self.height, self.pts)
    }
}

fn materialize_pixel_buffer(
    pixel_buffer: sys::CVPixelBufferRef,
    width: u32,
    height: u32,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let vt = sys::vtable().map_err(|e| Error::other(format!("videotoolbox: {e}")))?;
    let ret = unsafe { (vt.cv_pb_lock)(pixel_buffer, K_CV_PIXEL_BUFFER_LOCK_FLAGS_READ_ONLY) };
    if ret != 0 {
        return Err(Error::other(format!(
            "CVPixelBufferLockBaseAddress: {}",
            sys::describe_os_status(ret)
        )));
    }

    let result = (|| {
        let width = width as usize;
        let height = height as usize;
        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);

        let y_ptr = unsafe { (vt.cv_pb_get_base_of_plane)(pixel_buffer, 0) } as *const u8;
        let y_stride = unsafe { (vt.cv_pb_get_bpr_of_plane)(pixel_buffer, 0) };
        let y_height = unsafe { (vt.cv_pb_get_height_of_plane)(pixel_buffer, 0) };
        let uv_ptr = unsafe { (vt.cv_pb_get_base_of_plane)(pixel_buffer, 1) } as *const u8;
        let uv_stride = unsafe { (vt.cv_pb_get_bpr_of_plane)(pixel_buffer, 1) };
        let uv_height = unsafe { (vt.cv_pb_get_height_of_plane)(pixel_buffer, 1) };

        if y_ptr.is_null() || uv_ptr.is_null() {
            return Err(Error::invalid(
                "VideoToolbox CVPixelBuffer is missing an NV12 plane",
            ));
        }

        let mut y_data = vec![0u8; width * height];
        let mut u_data = vec![0u8; chroma_w * chroma_h];
        let mut v_data = vec![0u8; chroma_w * chroma_h];

        for row in 0..y_height.min(height) {
            let row_len = width.min(y_stride);
            let src = unsafe { std::slice::from_raw_parts(y_ptr.add(row * y_stride), row_len) };
            let dst = row * width;
            y_data[dst..dst + row_len].copy_from_slice(src);
        }

        for row in 0..uv_height.min(chroma_h) {
            let row_len = (chroma_w * 2).min(uv_stride);
            let src = unsafe { std::slice::from_raw_parts(uv_ptr.add(row * uv_stride), row_len) };
            let dst = row * chroma_w;
            for col in 0..chroma_w {
                u_data[dst + col] = if col * 2 < row_len { src[col * 2] } else { 128 };
                v_data[dst + col] = if col * 2 + 1 < row_len {
                    src[col * 2 + 1]
                } else {
                    128
                };
            }
        }

        Ok(VideoFrame {
            pts,
            planes: vec![
                VideoPlane {
                    stride: width,
                    data: y_data,
                },
                VideoPlane {
                    stride: chroma_w,
                    data: u_data,
                },
                VideoPlane {
                    stride: chroma_w,
                    data: v_data,
                },
            ],
        })
    })();

    unsafe { (vt.cv_pb_unlock)(pixel_buffer, 0) };
    result
}

// ─────────────────────────── Callback state ───────────────────────────────────

struct CallbackState {
    frames: VecDeque<FrameLease>,
    error: Option<String>,
}

impl CallbackState {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            frames: VecDeque::new(),
            error: None,
        }))
    }
}

unsafe extern "C" fn decomp_callback(
    output_callback_ref_con: *mut c_void,
    _source_frame_ref_con: *mut c_void,
    status: i32,
    _info_flags: u32,
    image_buffer: sys::CVImageBufferRef,
    presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    let state_ptr = output_callback_ref_con as *const Mutex<CallbackState>;
    let state = unsafe { &*state_ptr };
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return,
    };

    if status != K_OS_STATUS_NO_ERROR {
        guard.error = Some(format!(
            "VT decode callback: OSStatus {}",
            sys::describe_os_status(status)
        ));
        return;
    }
    if image_buffer.is_null() {
        return;
    }

    let vt = match sys::vtable() {
        Ok(v) => v,
        Err(e) => {
            guard.error = Some(format!("vtable in callback: {e}"));
            return;
        }
    };

    let width = unsafe { (vt.cv_pb_get_width)(image_buffer) };
    let height = unsafe { (vt.cv_pb_get_height)(image_buffer) };
    let Ok(width) = u32::try_from(width) else {
        guard.error = Some("VideoToolbox decoded width exceeds u32".into());
        return;
    };
    let Ok(height) = u32::try_from(height) else {
        guard.error = Some("VideoToolbox decoded height exceeds u32".into());
        return;
    };

    // Submission wraps packet PTS in a timescale-1 000 000 CMTime, and VT
    // returns the same value in presentation order.
    let pts = presentation_time_stamp
        .is_valid()
        .then_some(presentation_time_stamp.value);

    // The callback's image_buffer reference is borrowed. Retain it before the
    // callback returns; VideoToolboxVideoFrameStorage releases it on final drop.
    unsafe { (vt.cf_retain)(image_buffer) };
    guard
        .frames
        .push_back(FrameLease::from_hardware_video(HardwareVideoFrame::new(
            VideoToolboxVideoFrameStorage {
                pixel_buffer: image_buffer,
                width,
                height,
                pts,
            },
        )));
}

// ─────────────────────────── Session creation ─────────────────────────────────

/// Returns (session, retained_fmt_desc).
/// Caller must call `cf_release` on the returned `fmt_desc` when done — but the
/// session itself retains it, so the caller must retain it separately if they
/// want to keep using it after releasing.
///
/// `hw_mode` (from `options["hardware"]`) selects the optional
/// video-decoder-specification dictionary
/// (`kVTVideoDecoderSpecification_{Require,Enable}HardwareAcceleratedVideoDecoder`
/// per `VTDecompressionProperties.h`); `None` keeps VT's default policy.
fn create_vt_session(
    vt: &sys::Vtable,
    fmt_desc: sys::CMVideoFormatDescriptionRef,
    state: &Arc<Mutex<CallbackState>>,
    hw_mode: Option<HardwareMode>,
) -> Result<sys::VTDecompressionSessionRef> {
    let pixel_fmt_val = K_CV_PIXEL_FORMAT_420_YPCBCRi8_BI_PLANAR_VIDEO_RANGE as i32;
    let pixel_fmt_num = unsafe { sys::cf_number_i32(vt, pixel_fmt_val) };
    let pf_key = unsafe { sys::cf_string(vt, "CVPixelBufferPixelFormatTypeKey") };
    let keys: [*const c_void; 1] = [pf_key as *const c_void];
    let vals: [*const c_void; 1] = [pixel_fmt_num as *const c_void];

    let dest_attrs = unsafe {
        (vt.cf_dict_create)(
            std::ptr::null_mut(),
            keys.as_ptr(),
            vals.as_ptr(),
            1,
            vt.cf_type_dict_key_callbacks,
            vt.cf_type_dict_value_callbacks,
        )
    };

    let state_raw = Arc::as_ptr(state) as *mut c_void;
    let record = VTDecompressionOutputCallbackRecord {
        decomp_output_callback: decomp_callback,
        decomp_output_ref_con: state_raw,
    };

    let hw_spec = match hw_mode {
        Some(mode) => unsafe { build_hw_spec_dict(vt, mode, false) },
        None => std::ptr::null_mut(),
    };

    let mut session = std::ptr::null_mut();
    let status = unsafe {
        (vt.vt_decomp_create)(
            std::ptr::null_mut(),
            fmt_desc,
            hw_spec,
            dest_attrs,
            &record,
            &mut session,
        )
    };

    if !hw_spec.is_null() {
        unsafe { (vt.cf_release)(hw_spec) };
    }
    unsafe { (vt.cf_release)(dest_attrs) };
    unsafe { (vt.cf_release)(pixel_fmt_num) };
    unsafe { (vt.cf_release)(pf_key) };

    if status != K_OS_STATUS_NO_ERROR {
        Err(vt_error("VTDecompressionSessionCreate", status))
    } else {
        Ok(session)
    }
}

/// Submit AVCC-framed NAL units to a VT session.
/// `fmt_desc` must be the format description the session was created with.
fn submit_nal_units(
    vt: &sys::Vtable,
    session: sys::VTDecompressionSessionRef,
    fmt_desc: sys::CMVideoFormatDescriptionRef,
    nal_units: &[Vec<u8>],
    packet: &Packet,
    pts_counter: i64,
) -> Result<()> {
    if nal_units.is_empty() {
        return Ok(());
    }

    // Build AVCC payload (4-byte big-endian length prefix per NAL).
    let mut avcc: Vec<u8> = Vec::new();
    for nal in nal_units {
        let len = nal.len() as u32;
        avcc.extend_from_slice(&len.to_be_bytes());
        avcc.extend_from_slice(nal);
    }
    let avcc_len = avcc.len();

    // Create CMBlockBuffer with a copy of the data (custom allocator = kCFAllocatorNull
    // would work, but using a copied buffer is simpler and safe).
    let mut block_buf: sys::CMBlockBufferRef = std::ptr::null_mut();

    // Use CMBlockBufferCreateWithMemoryBlock. We pass a NULL memory block
    // which means VT allocates and copies the data itself.
    // To do a safe no-copy: we'd need to keep `avcc` alive until VT is done.
    // Instead, we allocate separately and let CF own it.
    //
    // kCFAllocatorDefault is NULL in the API.
    // kCFAllocatorNull = pointer to a global; we can't get it easily without headers.
    //
    // Simplest safe approach: allocate our own copy via raw malloc, pass it with
    // a NULL block allocator (which = default = malloc family), and CF will own it.
    // But we need CF to free it, which it won't if blockAllocator is NULL.
    //
    // Correct approach: pass the data pointer AND a non-null blockAllocator that
    // frees via `free()`. We'll approximate kCFAllocatorMalloc by using NULL
    // for the structure allocator and passing a malloc'd copy.

    // Allocate a copy using the system allocator that CF will free.
    let data_copy = unsafe {
        let ptr = libc_malloc(avcc_len);
        if ptr.is_null() {
            return Err(Error::other("malloc for CMBlockBuffer data failed"));
        }
        std::ptr::copy_nonoverlapping(avcc.as_ptr(), ptr as *mut u8, avcc_len);
        ptr
    };

    let status = unsafe {
        (vt.cm_block_create_with_mem)(
            std::ptr::null_mut(), // structure allocator = kCFAllocatorDefault
            data_copy,
            avcc_len,
            std::ptr::null_mut(), // block allocator = kCFAllocatorDefault = will free data_copy
            std::ptr::null(),     // custom block source
            0,                    // offset
            avcc_len,
            0, // flags
            &mut block_buf,
        )
    };

    if status != K_OS_STATUS_NO_ERROR {
        // Free our copy since CF won't.
        unsafe { libc_free(data_copy) };
        return Err(vt_error("CMBlockBufferCreateWithMemoryBlock", status));
    }

    let to_cmtime = |ticks: i64| {
        CMTime::make(
            packet
                .time_base
                .rescale(ticks, oxideav_core::TimeBase::MICROS),
            1_000_000,
        )
    };
    let timing = CMSampleTimingInfo {
        duration: packet.duration.map_or_else(CMTime::invalid, to_cmtime),
        presentation_time_stamp: packet
            .pts
            .map_or_else(|| CMTime::make(pts_counter, 1_000_000), to_cmtime),
        decode_time_stamp: packet.dts.map_or_else(CMTime::invalid, to_cmtime),
    };

    let mut sample_buf: sys::CMSampleBufferRef = std::ptr::null_mut();
    let status = unsafe {
        (vt.cm_sample_create_ready)(
            std::ptr::null_mut(),
            block_buf,
            fmt_desc, // must match the session's format description
            1,        // num samples
            1,        // num timing entries
            &timing,
            1,
            &avcc_len,
            &mut sample_buf,
        )
    };

    unsafe { (vt.cf_release)(block_buf) };

    if status != K_OS_STATUS_NO_ERROR {
        return Err(vt_error("CMSampleBufferCreateReady", status));
    }

    let dec_status = unsafe {
        (vt.vt_decomp_decode)(
            session,
            sample_buf,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };

    unsafe { (vt.cf_release)(sample_buf) };

    if dec_status != K_OS_STATUS_NO_ERROR {
        return Err(vt_error("VTDecompressionSessionDecodeFrame", dec_status));
    }

    Ok(())
}

// Minimal libc shim — just malloc/free.
unsafe fn libc_malloc(size: usize) -> *mut c_void {
    extern "C" {
        fn malloc(size: usize) -> *mut c_void;
    }
    unsafe { malloc(size) }
}

unsafe fn libc_free(ptr: *mut c_void) {
    extern "C" {
        fn free(ptr: *mut c_void);
    }
    unsafe { free(ptr) }
}

// ─────────────────────────── H.264 decoder ────────────────────────────────────

pub struct H264VtDecoder {
    codec_id: CodecId,
    session: sys::VTDecompressionSessionRef,
    /// Retained format description — used to create CMSampleBuffers.
    fmt_desc: sys::CMVideoFormatDescriptionRef,
    state: Arc<Mutex<CallbackState>>,
    sps_list: Vec<Vec<u8>>,
    pps_list: Vec<Vec<u8>>,
    au_assembler: AnnexBAccessUnitAssembler,
    output_queue: VecDeque<FrameLease>,
    pts_counter: i64,
    flushed: bool,
    /// Hardware-acceleration policy from `options["hardware"]`.
    hw_mode: Option<HardwareMode>,
}

unsafe impl Send for H264VtDecoder {}

impl H264VtDecoder {
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        sys::vtable().map_err(|e| Error::unsupported(format!("videotoolbox: {e}")))?;
        Ok(Box::new(H264VtDecoder {
            codec_id: CodecId::new("h264"),
            session: std::ptr::null_mut(),
            fmt_desc: std::ptr::null_mut(),
            state: CallbackState::new(),
            sps_list: Vec::new(),
            pps_list: Vec::new(),
            au_assembler: AnnexBAccessUnitAssembler::default(),
            output_queue: VecDeque::new(),
            pts_counter: 0,
            flushed: false,
            hw_mode: params.options.get("hardware").and_then(parse_hardware_mode),
        }))
    }

    fn ensure_session(&mut self) -> Result<()> {
        if !self.session.is_null() {
            return Ok(());
        }
        if self.sps_list.is_empty() || self.pps_list.is_empty() {
            return Ok(());
        }

        let vt = sys::vtable().map_err(|e| Error::unsupported(format!("videotoolbox: {e}")))?;

        let param_ptrs: Vec<*const u8> = self
            .sps_list
            .iter()
            .chain(self.pps_list.iter())
            .map(|v| v.as_ptr())
            .collect();
        let param_sizes: Vec<usize> = self
            .sps_list
            .iter()
            .chain(self.pps_list.iter())
            .map(|v| v.len())
            .collect();

        let mut fmt_desc: sys::CMVideoFormatDescriptionRef = std::ptr::null_mut();
        let st = unsafe {
            (vt.cm_fmt_from_h264_params)(
                std::ptr::null_mut(),
                param_ptrs.len(),
                param_ptrs.as_ptr(),
                param_sizes.as_ptr(),
                4,
                &mut fmt_desc,
            )
        };
        if st != K_OS_STATUS_NO_ERROR {
            return Err(vt_error(
                "CMVideoFormatDescriptionCreateFromH264ParameterSets",
                st,
            ));
        }

        let session = create_vt_session(vt, fmt_desc, &self.state, self.hw_mode)?;

        // Retain fmt_desc for our own use (the session also retains it).
        unsafe { (vt.cf_retain)(fmt_desc) };

        self.session = session;
        self.fmt_desc = fmt_desc;

        // Release the creation reference — we hold our own retained copy.
        unsafe { (vt.cf_release)(fmt_desc) };

        Ok(())
    }

    fn pull_frames(&mut self) {
        if let Ok(mut g) = self.state.lock() {
            while let Some(f) = g.frames.pop_front() {
                self.output_queue.push_back(f);
            }
        }
    }

    fn release_session(&mut self) {
        if let Ok(vt) = sys::vtable() {
            if !self.session.is_null() {
                // A seek starts a new decode epoch. Finish anything VT can
                // still complete, then invalidate stale reference-picture state.
                unsafe {
                    (vt.vt_decomp_finish)(self.session);
                    (vt.vt_decomp_invalidate)(self.session);
                    (vt.cf_release)(self.session);
                }
                self.session = std::ptr::null_mut();
            }
            if !self.fmt_desc.is_null() {
                unsafe { (vt.cf_release)(self.fmt_desc) };
                self.fmt_desc = std::ptr::null_mut();
            }
        }
    }

    fn process_access_unit(&mut self, packet: &Packet) -> Result<()> {
        let mut vcl_nals: Vec<Vec<u8>> = Vec::new();
        let mut params_changed = false;

        for nal in annex_b_nals(&packet.data) {
            if nal.is_empty() {
                continue;
            }
            let nal_type = nal[0] & 0x1F;
            match nal_type {
                h264_nal::SPS => {
                    params_changed |=
                        self.sps_list.len() != 1 || self.sps_list[0].as_slice() != nal;
                    self.sps_list.clear();
                    self.sps_list.push(nal.to_vec());
                }
                h264_nal::PPS => {
                    params_changed |=
                        self.pps_list.len() != 1 || self.pps_list[0].as_slice() != nal;
                    self.pps_list.clear();
                    self.pps_list.push(nal.to_vec());
                }
                _ => vcl_nals.push(nal.to_vec()),
            }
        }

        if params_changed && !self.session.is_null() {
            self.release_session();
            self.clear_pending_output();
        }

        if !vcl_nals.is_empty() {
            self.ensure_session()?;
        }

        if !vcl_nals.is_empty() && !self.session.is_null() {
            let vt = sys::vtable().map_err(|e| Error::unsupported(format!("videotoolbox: {e}")))?;
            let ctr = self.pts_counter;
            submit_nal_units(vt, self.session, self.fmt_desc, &vcl_nals, packet, ctr)?;
            self.pts_counter += 1;
        }

        self.pull_frames();
        Ok(())
    }

    fn clear_pending_output(&mut self) {
        self.output_queue.clear();
        if let Ok(mut state) = self.state.lock() {
            state.frames.clear();
            state.error = None;
        }
    }
}

impl Drop for H264VtDecoder {
    fn drop(&mut self) {
        self.release_session();
    }
}

impl oxideav_core::Decoder for H264VtDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.flushed = false;

        if let Some(e) = self
            .state
            .lock()
            .ok()
            .and_then(|mut g| g.error.take().map(Error::other))
        {
            return Err(e);
        }

        // Container/PES packet boundaries are not H.264 access-unit boundaries.
        // Reassemble AUD-delimited Annex-B access units before handing them to
        // VideoToolbox, matching the other OxideAV hardware backends.
        let completed = self.au_assembler.push(packet)?;
        for access_unit in completed {
            self.process_access_unit(&access_unit)?;
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.receive_frame_lease()?.into_frame()
    }

    fn receive_frame_lease(&mut self) -> Result<FrameLease> {
        if let Some(frame) = self.output_queue.pop_front() {
            return Ok(frame);
        }
        Err(if self.flushed {
            Error::Eof
        } else {
            Error::NeedMore
        })
    }

    fn flush(&mut self) -> Result<()> {
        if let Some(access_unit) = self.au_assembler.flush() {
            self.process_access_unit(&access_unit)?;
        }
        if !self.session.is_null() {
            if let Ok(vt) = sys::vtable() {
                unsafe { (vt.vt_decomp_finish)(self.session) };
            }
        }
        self.pull_frames();
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Flush is an end-of-stream drain; a seek starts a new decode epoch.
        // Retain the parameter sets, but discard the old VT session and all
        // queued output so reference pictures cannot cross the discontinuity.
        self.au_assembler.reset();
        self.release_session();
        self.clear_pending_output();
        self.pts_counter = 0;
        self.flushed = false;
        Ok(())
    }
}

// ─────────────────────────── HEVC decoder ─────────────────────────────────────

pub struct HevcVtDecoder {
    codec_id: CodecId,
    session: sys::VTDecompressionSessionRef,
    fmt_desc: sys::CMVideoFormatDescriptionRef,
    state: Arc<Mutex<CallbackState>>,
    vps_list: Vec<Vec<u8>>,
    sps_list: Vec<Vec<u8>>,
    pps_list: Vec<Vec<u8>>,
    output_queue: VecDeque<FrameLease>,
    pts_counter: i64,
    flushed: bool,
    /// Hardware-acceleration policy from `options["hardware"]`.
    hw_mode: Option<HardwareMode>,
}

unsafe impl Send for HevcVtDecoder {}

impl HevcVtDecoder {
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        sys::vtable().map_err(|e| Error::unsupported(format!("videotoolbox: {e}")))?;
        Ok(Box::new(HevcVtDecoder {
            codec_id: CodecId::new("hevc"),
            session: std::ptr::null_mut(),
            fmt_desc: std::ptr::null_mut(),
            state: CallbackState::new(),
            vps_list: Vec::new(),
            sps_list: Vec::new(),
            pps_list: Vec::new(),
            output_queue: VecDeque::new(),
            pts_counter: 0,
            flushed: false,
            hw_mode: params.options.get("hardware").and_then(parse_hardware_mode),
        }))
    }

    fn ensure_session(&mut self) -> Result<()> {
        if !self.session.is_null() {
            return Ok(());
        }
        if self.sps_list.is_empty() || self.pps_list.is_empty() {
            return Ok(());
        }

        let vt = sys::vtable().map_err(|e| Error::unsupported(format!("videotoolbox: {e}")))?;

        let param_ptrs: Vec<*const u8> = self
            .vps_list
            .iter()
            .chain(self.sps_list.iter())
            .chain(self.pps_list.iter())
            .map(|v| v.as_ptr())
            .collect();
        let param_sizes: Vec<usize> = self
            .vps_list
            .iter()
            .chain(self.sps_list.iter())
            .chain(self.pps_list.iter())
            .map(|v| v.len())
            .collect();

        let mut fmt_desc: sys::CMVideoFormatDescriptionRef = std::ptr::null_mut();
        let st = unsafe {
            (vt.cm_fmt_from_hevc_params)(
                std::ptr::null_mut(),
                param_ptrs.len(),
                param_ptrs.as_ptr(),
                param_sizes.as_ptr(),
                4,
                std::ptr::null_mut(),
                &mut fmt_desc,
            )
        };
        if st != K_OS_STATUS_NO_ERROR {
            return Err(vt_error(
                "CMVideoFormatDescriptionCreateFromHEVCParameterSets",
                st,
            ));
        }

        let session = create_vt_session(vt, fmt_desc, &self.state, self.hw_mode)?;

        unsafe { (vt.cf_retain)(fmt_desc) };
        self.session = session;
        self.fmt_desc = fmt_desc;
        unsafe { (vt.cf_release)(fmt_desc) };

        Ok(())
    }

    fn pull_frames(&mut self) {
        if let Ok(mut g) = self.state.lock() {
            while let Some(f) = g.frames.pop_front() {
                self.output_queue.push_back(f);
            }
        }
    }

    fn release_session(&mut self) {
        if let Ok(vt) = sys::vtable() {
            if !self.session.is_null() {
                unsafe {
                    (vt.vt_decomp_finish)(self.session);
                    (vt.vt_decomp_invalidate)(self.session);
                    (vt.cf_release)(self.session);
                }
                self.session = std::ptr::null_mut();
            }
            if !self.fmt_desc.is_null() {
                unsafe { (vt.cf_release)(self.fmt_desc) };
                self.fmt_desc = std::ptr::null_mut();
            }
        }
    }

    fn clear_pending_output(&mut self) {
        self.output_queue.clear();
        if let Ok(mut state) = self.state.lock() {
            state.frames.clear();
            state.error = None;
        }
    }
}

impl Drop for HevcVtDecoder {
    fn drop(&mut self) {
        self.release_session();
    }
}

impl oxideav_core::Decoder for HevcVtDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.flushed = false;

        // Surface (and clear) any error the async callback recorded since
        // the last call. `take()` keeps one bad frame from latching the
        // session into a permanent error state — VT decode errors are
        // per-frame, and the session remains usable for the next access
        // unit.
        if let Some(e) = self
            .state
            .lock()
            .ok()
            .and_then(|mut g| g.error.take().map(Error::other))
        {
            return Err(e);
        }

        let mut vcl_nals: Vec<Vec<u8>> = Vec::new();
        let mut params_changed = false;

        for nal in annex_b_nals(&packet.data) {
            if nal.len() < 2 {
                continue;
            }
            let nal_type = (nal[0] >> 1) & 0x3F;
            match nal_type {
                hevc_nal::VPS => {
                    params_changed |=
                        self.vps_list.len() != 1 || self.vps_list[0].as_slice() != nal;
                    self.vps_list.clear();
                    self.vps_list.push(nal.to_vec());
                }
                hevc_nal::SPS => {
                    params_changed |=
                        self.sps_list.len() != 1 || self.sps_list[0].as_slice() != nal;
                    self.sps_list.clear();
                    self.sps_list.push(nal.to_vec());
                }
                hevc_nal::PPS => {
                    params_changed |=
                        self.pps_list.len() != 1 || self.pps_list[0].as_slice() != nal;
                    self.pps_list.clear();
                    self.pps_list.push(nal.to_vec());
                }
                _ => vcl_nals.push(nal.to_vec()),
            }
        }

        if params_changed && !self.session.is_null() {
            self.release_session();
            self.clear_pending_output();
        }

        if !vcl_nals.is_empty() {
            self.ensure_session()?;
        }

        if !vcl_nals.is_empty() && !self.session.is_null() {
            let vt = sys::vtable().map_err(|e| Error::unsupported(format!("videotoolbox: {e}")))?;
            let ctr = self.pts_counter;
            submit_nal_units(vt, self.session, self.fmt_desc, &vcl_nals, packet, ctr)?;
            self.pts_counter += 1;
            unsafe { (vt.vt_decomp_finish)(self.session) };
        }

        self.pull_frames();
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.receive_frame_lease()?.into_frame()
    }

    fn receive_frame_lease(&mut self) -> Result<FrameLease> {
        if let Some(frame) = self.output_queue.pop_front() {
            return Ok(frame);
        }
        Err(if self.flushed {
            Error::Eof
        } else {
            Error::NeedMore
        })
    }

    fn flush(&mut self) -> Result<()> {
        if !self.session.is_null() {
            if let Ok(vt) = sys::vtable() {
                unsafe { (vt.vt_decomp_finish)(self.session) };
            }
        }
        self.pull_frames();
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.release_session();
        self.clear_pending_output();
        self.pts_counter = 0;
        self.flushed = false;
        Ok(())
    }
}
