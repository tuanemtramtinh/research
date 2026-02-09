# Use Case Description Report

## 1. Use Case Name  
**Upload Dataset**

## 2. Description  
Depositors submit new datasets to the archive, initiating the record creation and metadata harvesting process.

## 3. Primary Actor  
**Depositor**

## 4. Problem Domain Context  
Create a unified, web‑based research data archive that allows depositors to upload, version, and manage datasets—integrating with Pure, VREs, and external repositories—while automatically harvesting and synchronising metadata, applying licenses, embargoes, and disposal policies; provide depositors with DOI minting, persistent URLs, and analytics (downloads, citations), and enable privileged access for collaborators and external users; expose a multilingual, searchable interface (via web and Primo) for data reusers to browse, preview, and cite datasets, including version comparison; enforce quality checks and required metadata for administrators, support bulk import/export and API (SWORD2) access, and integrate seamlessly with university systems (LDAP, CRIS, HCP storage) to support reporting, impact analysis, and compliance with funder and institutional policies.

## 5. Preconditions  
- The depositor is authenticated via the university LDAP system.  
- The depositor has an active user profile with the required permissions to create a new dataset.  
- The local network connection is stable and the web application is reachable.  
- The uploader has prepared the dataset files and accompanying metadata files (e.g., `metadata.xml`, `README.txt`) ready for submission.  

If no conditions apply, state: **None.**

## 6. Postconditions  
- A new dataset record is created in the archive with a unique identifier (e.g., a DOI).  
- All uploaded files are stored in the HCP storage system and are accessible via persistent URLs.  
- Metadata is harvested from the submitted files and/or external sources (Pure, VREs) and stored in the archive’s metadata repository.  
- Licenses, embargoes, and disposal policies are applied to the dataset.  
- Analytics counters for downloads and citations are initialized.  
- The dataset is indexed in the search interface (web and Primo).  
- Notifications are sent to the depositor and relevant collaborators.  

If none, state: **None.**

## 7. Main Flow  

1. **Depositor initiates upload** – The depositor logs into the portal and selects “Upload Dataset.”  
2. **System presents upload wizard** – The interface shows steps: (a) choose files, (b) enter metadata, (c) set license/embargo, (d) confirm.  
3. **Depositor selects files** – The depositor uploads one or more files (data, code, documentation).  
4. **System validates file types and size** – The system checks against allowed MIME types and size limits.  
5. **Depositor enters metadata** – The depositor fills mandatory fields (title, authors, description, keywords, etc.) and optionally provides a license and embargo date.  
6. **System validates metadata** – Required fields are verified; missing or invalid entries trigger warnings.  
7. **Depositor reviews and confirms** – The depositor reviews a summary page and confirms submission.  
8. **System creates dataset record** – A new record is created with a provisional identifier.  
9. **System persists files** – Uploaded files are stored in the HCP storage, and checksums are computed.  
10. **System applies policies** – Licenses, embargoes, and disposal policies are attached to the record.  
11. **System harvests metadata** – The system extracts additional metadata from files (e.g., from CSV headers, JSON schemas) and synchronises with external sources (Pure, VREs).  
12. **System mints DOI and URLs** – A DOI is minted via the DOI service, and persistent URLs are generated.  
13. **System indexes dataset** – The record is indexed for search visibility in web and Primo.  
14. **System initializes analytics** – Download and citation counters are set to zero.  
15. **System sends notifications** – Email/SMS notifications are sent to the depositor and collaborators.  
16. **Depositor receives confirmation** – The depositor sees a success page with the DOI and access links.  

## 8. Alternative Flows  

**AF1 – Partial Metadata Completion**  
- *Trigger:* Step 6 (metadata validation).  
- 6a. System detects missing optional fields.  
- 6b. System allows the depositor to skip or complete later.  

**AF2 – License Conflict**  
- *Trigger:* Step 6 (metadata validation).  
- 6c. System detects a license that conflicts with institutional policy.  
- 6d. System prompts the depositor to choose an alternative license or request approval.  

**AF3 – Embargo Expiration Management**  
- *Trigger:* Step 11 (metadata harvesting).  
- 11a. System schedules a background job to lift embargo at the specified date.  

## 9. Exceptions  

**E1 – File Upload Failure**  
- *Step:* 3 (file selection).  
- *Condition:* Network interruption or file exceeds size limit.  
- *Handling:* System displays an error message, logs the incident, and offers retry or cancel options.  

**E2 – Metadata Validation Error**  
- *Step:* 6 (metadata validation).  
- *Condition:* Mandatory field missing or data format incorrect.  
- *Handling:* System highlights the problematic fields, provides guidance, and prevents progress until resolved.  

**E3 – DOI Minting Failure**  
- *Step:* 12 (DOI minting).  
- *Condition:* External DOI service unavailable or rate‑limited.  
- *Handling:* System queues the request, retries after back‑off, and logs the failure. The depositor is notified of the delay.  

**E4 – Policy Rejection**  
- *Step:* 10 (policy application).  
- *Condition:* Applied license or embargo violates institutional or funder policy.  
- *Handling:* System rejects the submission, informs the depositor with reasons, and provides options to modify the policy or request an exception.  

**E5 – Storage Capacity Exceeded**  
- *Step:* 9 (file persistence).  
- *Condition:* HCP storage quota exceeded.  
- *Handling:* System aborts the upload, notifies the depositor, and offers to free space or split the upload.  

**E6 – External Source Unreachable**  
- *Step:* 11 (metadata harvest).  
- *Condition:* Connection to Pure or VRE fails.  
- *Handling:* System logs the error, continues with local metadata, and informs the depositor that external metadata will be updated when available.  

**E7 – Authentication Expired**  
- *Step:* Any step after 1.  
- *Condition:* User session times out.  
- *Handling:* System redirects to login, preserves unsaved progress when possible, and resumes after re‑authentication.  

**E8 – System Crash**  
- *Step:* Any step.  
- *Condition:* Unexpected exception or database failure.  
- *Handling:* System displays a generic error page, logs the stack trace, sends an alert to the support team, and rolls back the transaction to maintain consistency.