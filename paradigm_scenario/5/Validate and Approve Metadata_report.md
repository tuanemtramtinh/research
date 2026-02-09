# Use Case Description Report  
**Use Case Name:** Validate and Approve Metadata  

## 1. Description  
Administrators enforce quality checks, ensure required metadata is present, and approve records before publication.  

## 2. Primary Actor  
Administrator  

## 3. Problem Domain Context  
Create a unified, web‑based research data archive that allows depositors to upload, version, and manage datasets—integrating with Pure, VREs, and external repositories—while automatically harvesting and synchronising metadata, applying licenses, embargoes, and disposal policies; provide depositors with DOI minting, persistent URLs, and analytics (downloads, citations), and enable privileged access for collaborators and external users; expose a multilingual, searchable interface (via web and Primo) for data reusers to browse, preview, and cite datasets, including version comparison; enforce quality checks and required metadata for administrators, support bulk import/export and API (SWORD2) access, and integrate seamlessly with university systems (LDAP, CRIS, HCP storage) to support reporting, impact analysis, and compliance with funder and institutional policies.  

## 4. Preconditions  
- The dataset record has been created and is in a “Pending Validation” state.  
- The Administrator is authenticated and has the *Metadata Approver* role.  

## 5. Postconditions  
- The dataset record is marked **Approved** and made available for publication.  
- Audit log entries record the approval action and timestamp.  

## 6. Main Flow  
1. **Administrator** logs into the archive management portal.  
2. **Administrator** navigates to the *Pending Validation* queue.  
3. **Administrator** selects a dataset record to validate.  
4. The system **retrieves** the current metadata record and displays it in the validation interface.  
5. **Administrator** reviews the metadata against institutional quality rules (e.g., required fields, controlled vocabularies, embargo dates).  
6. The system **runs automatic checks** (schema validation, controlled vocabulary mapping, DOI syntax).  
7. If any **issues** are found, they are listed with severity and suggested fixes.  
8. **Administrator** reviews issue list and decides to **resolve** or **reject** the record.  
9. If resolving, **Administrator** edits the metadata inline or uploads a corrected file.  
10. The system **re‑validates** the edited metadata automatically.  
11. Once **all issues are resolved**, **Administrator** clicks **Approve**.  
12. The system **updates** the record status to *Approved*, assigns a publication timestamp, and triggers downstream processes (DOI minting, persistent URL resolution, analytics initialization).  
13. The system **logs** the approval event and notifies relevant stakeholders (depositors, collaborators).  

## 7. Alternative Flows  
- **AF1 (Issues Unresolved)** – Begins at step 8.  
  1. **Administrator** chooses to **Reject** instead of approving.  
  2. The system **marks** the record as *Rejected* and sends a rejection notice to the depositor with feedback.  
- **AF2 (Automatic Corrections Applied)** – Begins at step 6.  
  1. The system automatically **applies** minor corrections (e.g., trimming whitespace, normalizing date formats).  
  2. The system **re‑validates** and proceeds to step 8 without manual edits.  

## 8. Exceptions  
- **E1 (Authentication Failure)** – Occurs at step 1.  
  - The system displays an error and redirects to the login page.  
- **E2 (Metadata Corrupt or Inaccessible)** – Occurs at step 4.  
  - The system logs the error, displays a message to the administrator, and offers to retry or abort.  
- **E3 (Validation Rule Violation Unrecoverable)** – Occurs during step 6.  
  - The system flags the record as *Blocked* and requires manual intervention by a senior administrator.  
- **E4 (Database Write Failure)** – Occurs at step 12.  
  - The system rolls back the status change, logs the failure, and alerts system admin via email.  

---