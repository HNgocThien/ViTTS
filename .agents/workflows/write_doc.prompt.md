---
description: thesis-level academic documentation Writing Task about TTS
---

# Documentation Writing Task

## Overview
You are writing technical documentation for the KLTN_TTS (Text-to-Speech Personal) project. This is a comprehensive thesis project implementing F5-TTS (Flow Matching-based TTS) with Vietnamese language support.

## Key Responsibilities

### 1. Content Requirements
- **Scope**: Write sections based on the documentation index structure in `docs/indexed.md`
- **Language**: Vietnamese (Tiếng Việt) with technical terminology clearly explained
- **Technical Depth**: Appropriate for thesis-level academic documentation
- **Audience**: Computer Science students and TTS researchers
- **Writing rules**: write by markdown format, and draw diagram by mermaid diagram
- **Reference document** Every knowledge/ mathematic formulate have to link, reference to puplic papers. 
- **Appendix**: Giải thích các thuật ngữ chuyên ngành, công thức toán, ..., vs dụ "what is prosody prediction?", người đọc là người chưa biết gì vẫn hiểu.
- **rule**: Tối ưu nội dung tôi viết theo hướng: ngắn gọn, Không dư lý thuyết cơ bản, Chỉ giữ lại phần phục vụ trực tiếp cho mô hình chính (F5-TTS), Logic mạch lạc, có dẫn dắt giữa các phần, Tránh “logic gap” (khái niệm xuất hiện phải được introduce trước), Viết theo phong cách academic, rõ ràng, không lan man, Không biến thành textbook, luôn hướng về bài toán chính: TTS và F5-TTS, Giữ lại các công thức quan trọng, nhưng không lạm dụng.


### 3. Documentation Maintenance

**After writing a new section, you MUST:**

1. **Update Status in Index**
   - Modify the corresponding row in `docs/indexed.md`
   - Change status from "Not Started" → "Started" → "Done"
   
2. **Link the Documentation**
   - Ensure the markdown file is properly referenced in the index table
   - Use relative paths: `[Section1.1.md](Section1.1.md)` or `[Section1.1.md](Section1.1.md#section-anchor)`

## Success Criteria
✓ Content is complete and technically accurate
✓ Status updated in `docs/indexed.md`
✓ File is linked and discoverable from the index
✓ Progress tracked in appropriate status files