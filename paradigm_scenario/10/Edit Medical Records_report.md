# Use Case Description Report

## 1. Use Case Name  
**Edit Medical Records**

## 2. Description  
Patient edits and manages their own medical records.

## 3. Primary Actor  
**Patient**

## 4. Problem Domain Context  
The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.

## 5. Preconditions  
- Patient is authenticated and authorized via DigiD or standard credentials.  
- Patient has at least one medical record in the system.  
- The system is online and accessible.  

## 6. Postconditions  
- The patient’s medical record is updated with the edits or additions made.  
- Audit trail entry is created for the edit action.  
- Relevant notifications (e.g., to assigned doctors) are generated if required.  

## 7. Main Flow  
1. **Patient** logs into the system using DigiD or credentials.  
2. **System** verifies authentication and displays the patient dashboard.  
3. **Patient** navigates to the “Medical Records” section.  
4. **System** lists all medical records with options to view, edit, or add new entries.  
5. **Patient** selects a record to edit.  
6. **System** presents the record details in an editable form.  
7. **Patient** modifies fields (e.g., diagnosis, medications, notes).  
8. **Patient** clicks “Save” or “Submit”.  
9. **System** validates input data and checks for conflicts or required fields.  
10. **System** saves changes to the database and writes an audit log entry.  
11. **System** displays a confirmation message to the patient.  
12. **System** updates any linked external device data if necessary.  
13. **System** sends notifications to relevant stakeholders (e.g., assigned doctors) if the record change requires review.  
14. **Patient** returns to the dashboard or continues editing other records.

## 8. Alternative Flows  
- **A1 – Adding a New Record** (branches from step 4):  
  1. **Patient** clicks “Add New Record”.  
  2. **System** presents a blank record form.  
  3. **Patient** fills in the details and saves.  
  4. **System** validates, saves, logs, confirms, and notifies stakeholders.  

- **A2 – Linking External Device Data** (branches from step 12):  
  1. **Patient** chooses to link external device data.  
  2. **System** prompts for device authorization and data import.  
  3. **Patient** authorizes and selects data to import.  
  4. **System** imports, validates, and associates data with the record.  

## 9. Exceptions  
- **E1 – Authentication Failure** (step 1):  
  1. System displays error message and prompts re‑login.  

- **E2 – Insufficient Permissions** (any step requiring edit):  
  1. System denies action and shows “Access Denied” message.  

- **E3 – Validation Error** (step 9):  
  1. System highlights problematic fields, shows error messages, and prevents save until corrected.  

- **E4 – System Downtime** (any step):  
  1. System displays “Service Unavailable” banner and queues request for retry.  

- **E5 – Data Conflict** (step 9):  
  1. System detects concurrent modification, offers merge options or abort.  

- **E6 – Notification Failure** (step 13):  
  1. System logs failure, retries, and informs patient of pending notification.