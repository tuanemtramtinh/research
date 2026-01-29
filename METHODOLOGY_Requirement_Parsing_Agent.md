# Methodology: Requirement Parsing Agent (RPA)

## 1. Tổng quan

**Requirement Parsing Agent (RPA)** là một agent xử lý yêu cầu phần mềm dựa trên đồ thị trạng thái (StateGraph), có nhiệm vụ phân tích văn bản user story dạng *“As a [actor], I want to [action], so that [benefit]”* và sinh ra mô hình Use Case dạng UML, bao gồm: danh sách actor chuẩn hóa, các use case với tên và mô tả, cùng quan hệ «include» và «extend» giữa các use case.

Quy trình xử lý được tổ chức thành **hai nhánh song song** (Actor pipeline và Use Case pipeline), **hội tụ** tại bước gom cặp actor–use case và phân cụm ngữ nghĩa, sau đó lần lượt qua tinh chỉnh cụm, đặt tên use case và phát hiện quan hệ.

---

## 2. Đầu vào và tiền xử lý

- **Đầu vào:** Chuỗi văn bản `requirement_text` chứa các user story, mỗi câu ứng với một dòng (hoặc một câu sau khi tách).
- **Tiền xử lý:** Văn bản được tách thành danh sách câu `sentences` (ví dụ theo ký tự xuống dòng). Mô hình ngôn ngữ (LLM) và công cụ NLP được nạp sẵn để dùng trong các bước sau.

---

## 3. Pipeline trích xuất Actor (Branch 1)

### 3.1 Trích xuất actor bằng regex (`find_actors`)

- **Mục tiêu:** Thu thập tất cả actor được nhắc trong mẫu *“As a/an/the [actor]”*.
- **Cách làm:**
  - Dùng biểu thức chính quy cố định: `As\s+(?:a|an|the)\s+([^,]+)`.
  - Với mỗi câu, lấy nhóm thứ nhất làm tên actor, chuẩn hóa chữ thường và ghi nhận các chỉ số câu xuất hiện.
  - **Lọc actor hệ thống:** Các actor chứa từ khóa *system, software, application, platform, service, backend, server* bị loại để chỉ giữ actor nghiệp vụ (user, admin, customer, …).
- **Kết quả:** Danh sách `raw_actors`, mỗi phần tử gồm `actor` (tên) và `sentence_idx` (các chỉ số câu).

### 3.2 Gộp actor đồng nghĩa (`synonym_check`)

- **Mục tiêu:** Gộp các cách gọi cùng một vai trò (ví dụ “user” và “customer” trong ngữ cảnh tương đương) thành một tên chuẩn.
- **Cách làm:** Gọi LLM với structured output (schema `CanonicalActorList`), đưa vào danh sách tên actor từ bước trước. Yêu cầu LLM chỉ trả về các tên chuẩn (canonical), ưu tiên tên có sẵn trong danh sách đầu vào, không thêm actor mới. Hợp nhất `sentence_idx` theo tên chuẩn.
- **Kết quả:** Danh sách `actors` – các actor chuẩn không trùng nghĩa.

### 3.3 Tìm alias cho actor (`find_aliases`)

- **Mục tiêu:** Với mỗi actor chuẩn, tìm các cách gọi thay thế (alias) **chỉ xuất hiện ở vị trí “As a [X]” ở đầu câu**.
- **Cách làm:** Gửi cho LLM danh sách actor chuẩn và toàn bộ câu (có đánh số). Yêu cầu LLM chỉ coi alias là từ/cụm từ nằm ở vị trí “As a [X]”, không coi từ trong phần thân câu (ví dụ “user” trong “log user activities”) là alias. Structured output theo schema `ActorAliasList`.
- **Kết quả:** Danh sách `actor_results`, mỗi phần tử gồm `actor`, `aliases` (mỗi alias kèm danh sách `sentences`), và `sentence_idx`.

---

## 4. Pipeline trích xuất Use Case (Branch 2)

### 4.1 Trích xuất use case bằng NLP (`find_usecases`)

- **Mục tiêu:** Từ mỗi câu, trích các “hành động mong muốn” tương ứng mẫu *“I want to [verb phrase]”* để dùng làm use case thô.
- **Cách làm:**
  - Dùng spaCy (mô hình `en_core_web_lg`) để phân tích cú pháp (dependency parsing).
  - Tìm token có `lemma_ == "want"`, rồi tìm con `dep_ == "xcomp"` và `pos_ in {VERB, AUX}` (bỏ qua dạng V-ing).
  - Từ xcomp, gọi `_find_main_verb` để lấy động từ chính (xử lý các cấu trúc kiểu “be adjective to [verb]”, “be V_3 to [verb]”).
  - Dùng `_get_verb_phrase` để lấy cả cụm động từ (verb + dobj, prep/pobj, prt, acomp, advmod, compound, amod), cắt bỏ mệnh đề “so that” nếu có.
  - Với mỗi động từ chính, gọi `_get_all_conj` để thu thập tất cả động từ trong chuỗi liên hợp (conj), và trích verb phrase cho từng động từ đó.
- **Kết quả:** Từ điển `raw_usecases` ánh xạ `sentence_idx → [danh sách use case chuỗi]`.

### 4.2 Tinh chỉnh use case bằng LLM (`refine_usecases`)

- **Mục tiêu:** Chuyển use case thô thành tên use case kiểu UML: ngắn gọn, dạng “động từ + bổ ngữ”, một use case một mục tiêu, không chứa “so that”, điều kiện hay chi tiết triển khai.
- **Cách làm:** Gửi LLM từng nhóm (theo `sentence_idx`) use case thô và toàn bộ câu. LLM trả về structured output (`UsecaseRefinementResponse`) gồm: `original`, `refined`, `added` (use case thiếu nhưng có trong câu), và `reasoning`. Chỉ bổ sung use case khi chúng **được nêu rõ** trong câu, không suy diễn thêm.
- **Kết quả:** Danh sách `refined_usecases`, mỗi phần tử có `sentence_idx`, `original`, `refined`, `added`.

---

## 5. Hội tụ: Ghép Actor–Use Case và phân cụm (Grouping)

### 5.1 Xây dựng cặp (actor, use case)

- **Điều kiện:** Chỉ chạy khi đã có đủ `actor_results` và `refined_usecases` (hai nhánh đã chạy xong).
- **Cách làm:**
  - Xây bảng tra `sentence_idx → [các actor]` từ `actor_results` (cả tên chuẩn và alias cùng tập `sentence_idx`/`sentences`).
  - Với mỗi phần tử trong `refined_usecases`, lấy `refined + added` làm danh sách use case. Với mỗi (sentence_idx, use case), ghép với mọi actor thuộc câu đó thành cặp (actor, use case).
  - Mỗi cặp được lưu dạng chuỗi `"<actor> <usecase>"` để embed, và kèm metadata: `actor`, `usecase`, `sentence_idx`, `original_sentence`.

### 5.2 Embedding và phân cụm K-Means

- **Embedding:** Dùng mô hình `text-embedding-3-large` (OpenAI) để biểu diễn từng chuỗi `"<actor> <usecase>"` thành vector.
- **Chọn số cụm:** Thử lần lượt `k = 2, 3, …, n-1` (n = số cặp), mỗi k chạy K-Means (random_state=42, n_init=10) và tính **silhouette score**. Chọn `k` có silhouette lớn nhất làm `best_k`.
- **Phân cụm cuối:** Chạy K-Means với `k = best_k`, gán mỗi cặp vào một cụm; mỗi cụm lưu danh sách phần tử dạng `{actor, usecase, sentence_idx, original_sentence}`.
- **Kết quả:** `user_story_clusters` – danh sách cụm, mỗi cụm có `cluster_id` và `user_stories`.

---

## 6. Tinh chỉnh cụm bằng LLM (`refine_clustering`)

- **Mục tiêu:** Sửa các cụm do K-Means tạo ra khi chúng gộp nhầm theo ngữ nghĩa bề mặt nhưng khác mục tiêu nghiệp vụ (ví dụ: đăng nhập vs quản lý tài khoản, xem vs xuất báo cáo).
- **Nguyên tắc:** Một use case UML tương ứng **một mục tiêu rời rạc** cho actor; tách bạch Identity/Authentication, Administration/CRUD, Transaction/Browsing.
- **Cách làm:** Gửi LLM toàn bộ cụm hiện tại và danh sách item (sentence_idx, actor, usecase). Yêu cầu LLM gán lại mỗi item vào `target_cluster_id` theo quy tắc hướng mục tiêu, tách theo ngữ cảnh actor và nghiệp vụ. Structured output theo `RefineClusteringResponse`. Nhóm lại theo `target_cluster_id` để tạo `user_story_clusters` mới (nếu thiếu gán thì giữ `cluster_id` cũ).
- **Kết quả:** Cập nhật `user_story_clusters` sau tinh chỉnh.

---

## 7. Đặt tên Use Case (`name_usecases`)

- **Mục tiêu:** Với mỗi cụm, sinh **một tên use case** và **một mô tả ngắn** đủ để dùng trong sơ đồ Use Case UML.
- **Cách làm:** Gửi LLM thông tin từng cụm: tập actor, danh sách action (usecase thô), và tối đa 3 câu mẫu. Yêu cầu đặt tên theo chuẩn verb–noun, 1–4 từ, theo các mẫu có sẵn (ví dụ: login/register/logout/reset password → “Authenticate”; CRUD → “Manage [Resource]”; xem/duyệt → “View/Browse [Data]”). Structured output theo `UseCaseNamingResponse` (cluster_id, usecase_name, description).
- **Đầu ra cấu trúc:** Tạo danh sách đối tượng `UseCase`, mỗi phần tử gồm: `id`, `name`, `description`, `participating_actors` (từ các story trong cụm), `user_stories` (chuyển từ `user_story_clusters` sang kiểu `UserStoryItem`), `relationships` (ban đầu rỗng).

---

## 8. Phát hiện quan hệ «include» / «extend» (`find_include_extend`)

- **Mục tiêu:** Xác định quan hệ «include» (bắt buộc) và «extend» (tùy chọn) giữa các use case và gắn vào từng `UseCase`.
- **Định nghĩa trong prompt:**
  - **«include»:** Use case A luôn cần thực hiện use case B (ví dụ: “Checkout” includes “Validate cart”). Từ khóa gợi ý: “must”, “requires”, “needs to”, “first”, “then”.
  - **«extend»:** Use case A có thể mở rộng use case B trong một số trường hợp (ví dụ: “Apply discount” extends “Checkout”). Từ khóa gợi ý: “optionally”, “can also”, “if”, “when”, “may”.
- **Cách làm:** Một lần gọi LLM với danh sách use case (tên, mô tả, ví dụ câu) và danh sách tên chính xác. Yêu cầu chỉ xuất quan hệ có căn cứ từ mô tả/ví dụ, không bịa; source/target phải trùng tên use case trong danh sách; không self-include/self-extend. Structured output theo `UseCaseRelationshipResponse` (source_use_case, relationship_type, target_use_case, reasoning).
- **Gắn vào UseCase:** Duyệt từng use case, với mỗi quan hệ có `source_use_case` trùng tên use case đó thì thêm vào `relationships` dạng `UseCaseRelationship(type, target_use_case)`.

---

## 9. Đầu ra cuối cùng

RPA trả về cấu trúc (ví dụ `RpaState`) gồm:

- **requirement_text:** Văn bản user story gốc.
- **actors:** Danh sách actor chuẩn (sau synonym check).
- **actor_aliases:** Danh sách `ActorResult` (actor chuẩn kèm alias và sentence indices).
- **use_cases:** Danh sách `UseCase` với `name`, `description`, `participating_actors`, `user_stories`, và `relationships` («include»/«extend»).

---

## 10. Sơ đồ luồng xử lý (tóm tắt)

```
                    [requirement_text → sentences]
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   ┌─────────────────┐                           ┌─────────────────┐
   │ find_actors     │ (regex + filter system)   │ find_usecases   │ (spaCy)
   └────────┬────────┘                           └────────┬────────┘
            ▼                                               ▼
   ┌─────────────────┐                           ┌─────────────────┐
   │ synonym_check   │ (LLM canonical names)     │ refine_usecases │ (LLM)
   └────────┬────────┘                           └────────┬────────┘
            ▼                                               ▼
   ┌─────────────────┐                                    │
   │ find_aliases    │ (LLM, "As a [X]" only)             │
   └────────┬────────┘                                    │
            └───────────────────────┬─────────────────────┘
                                    ▼
                          ┌─────────────────┐
                          │ grouping        │ (pairs → embed → K-Means, silhouette)
                          └────────┬────────┘
                                    ▼
                          ┌─────────────────┐
                          │ refine_clustering│ (LLM split/merge by goal)
                          └────────┬────────┘
                                    ▼
                          ┌─────────────────┐
                          │ name_usecases   │ (LLM name + description per cluster)
                          └────────┬────────┘
                                    ▼
                          ┌─────────────────┐
                          │ find_include_   │ (LLM «include»/«extend»)
                          │ extend          │
                          └────────┬────────┘
                                    ▼
                          [use_cases, actors, actor_aliases]
```

---

## 11. Công nghệ và phụ thuộc chính

- **Đồ thị luồng:** LangGraph `StateGraph` với `GraphState` (TypedDict).
- **LLM:** Mô hình chat (ví dụ OpenAI) với structured output (Pydantic).
- **NLP:** spaCy `en_core_web_lg` cho dependency parsing và trích verb phrase.
- **Embedding:** OpenAI `text-embedding-3-large`.
- **Phân cụm:** scikit-learn `KMeans`, đánh giá bằng `silhouette_score`.

Phương pháp trên cho phép Requirement Parsing Agent từ user story thô tự động tạo ra mô hình Use Case có cấu trúc, phù hợp dùng làm đầu vào cho các bước mô hình hóa hoặc sinh sơ đồ UML tiếp theo.
