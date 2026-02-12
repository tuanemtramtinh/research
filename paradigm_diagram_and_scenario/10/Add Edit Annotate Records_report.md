# Use Case Description Report

## 1. Use Case Name
Add Edit Annotate Records

## 2. Description
Doctor adds, edits, and annotates patient records

## 3. Primary Actor
Doctor

## 4. Problem Domain Context
The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.

## 5. Preconditions
- The doctor is authenticated and authorized to access the patient’s record.  
- The patient record exists and is not locked for editing by another user.  
- The system is online and all required services (e.g., database, authentication, audit log) are operational.

## 6. Postconditions
- The patient record is updated with the new or edited information.  
- Any annotations are stored and associated with the correct record version and timestamp.  
- Audit logs capture the action, actor, and affected data.  
- Relevant stakeholders (e.g., linked specialists, lab technicians) receive notifications if configured.

## 7. Main Flow
1. **Doctor** selects a patient record from the dashboard.  
2. System verifies the doctor’s permission to view the record.  
3. **Doctor** chooses to add a new record entry or edit an existing one.  
4. System presents the record form with current data pre‑filled (for edit) or blank (for add).  
5. **Doctor** enters or modifies the required fields (diagnosis, treatment notes, medications, etc.).  
6. **Doctor** clicks “Annotate” to add a comment or highlight a section.  
7. System displays an annotation editor; **Doctor** types the annotation and assigns a visibility level.  
8. **Doctor** reviews changes and clicks “Save”.  
9. System validates data integrity and checks for conflicts (e.g., concurrent edits).  
10. System writes changes to the database, increments the record version, and records audit details.  
11. System triggers any configured notifications to related stakeholders.  
12. System confirms success to the **Doctor** with a message and updates the UI.

## 8. Alternative Flows
- **AF1 (Step 3 – Add/Edit)**  
  3a. **Doctor** selects “Add External Device Data” instead of a standard record.  
  3b. System initiates device data retrieval workflow.  
  3c. **Doctor** confirms linkage; system attaches device data to the record.  
  3d. Return to Step 8 for saving.

- **AF2 (Step 6 – Annotate)**  
  6a. **Doctor** opts for a “Link Specialist” annotation.  
  6b. System opens specialist selection dialog.  
  6c. **Doctor** chooses specialist; system creates a referral link in the record.  
  6d. Return to Step 8 for saving.

## 9. Exceptions
- **E1 (Step 2 – Permission Check)**  
  1. System denies access; displays an error message “You do not have permission to view this record.”  
  2. Flow terminates; audit log records the denied attempt.

- **E2 (Step 9 – Data Validation)**  
  1. Validation fails (e.g., required field missing).  
  2. System highlights errors and prompts the **Doctor** to correct them.  
  3. Flow returns to Step 5 or 6 as appropriate.

- **E3 (Step 9 – Conflict Detection)**  
  1. Concurrent edit detected.  
  2. System notifies the **Doctor** of the conflict, shows the latest version, and offers options to override or merge.  
  3. If overridden, system proceeds to Step 10; otherwise, flow returns to Step 5.

- **E4 (Step 10 – Database Write Failure)**  
  1. System encounters a persistence error.  
  2. Displays “Save failed. Please try again later.”  
  3. Logs the error and rolls back any partial changes.  
  4. Flow terminates; no changes are committed.

- **E5 (Step 11 – Notification Failure)**  
  1. Notification service is unavailable.  
  2. System logs the failure and retries asynchronously.  
  3. Does not affect the main record save; user is informed that notifications may be delayed.