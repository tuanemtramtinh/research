# Use Case Description Report

1. **Use Case Name:** View Medical Records  
2. **Description:** Patient views their own medical records and linked device data.  
3. **Primary Actor:** Patient  
4. **Problem Domain Context:**  
   The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.  
5. **Preconditions:**  
   - The patient is authenticated and authorized via DigiD or system credentials.  
   - The patient's medical record exists in the system.  
   - Relevant external device data is linked to the patient's record.  
6. **Postconditions:**  
   - The patient has viewed the requested medical records and device data.  
   - System logs the view action for audit purposes.  
   - No changes to the data are made unless the patient initiates an edit.  
7. **Main Flow:**  
   1. **Patient** logs into the portal using DigiD or system credentials.  
   2. The system validates credentials and establishes a secure session.  
   3. **Patient** navigates to the “My Medical Records” section.  
   4. The system retrieves the patient’s medical record identifiers and displays a list of available records (e.g., visits, prescriptions, lab results).  
   5. **Patient** selects a specific record to view.  
   6. The system fetches the record details, including attached documents, notes, and linked device data.  
   7. The system renders the record in the user interface, highlighting any annotations or alerts.  
   8. **Patient** scrolls through the record, optionally expanding sections (e.g., lab results, imaging).  
   9. The system continuously updates the view with any real‑time data from linked devices (e.g., glucose monitor).  
   10. **Patient** closes the record or returns to the list.  
   11. The system records the view event with timestamp, device ID, and user ID for audit.  
8. **Alternative Flows:**  
   1. **Patient** attempts to view a record that is not yet available (e.g., pending lab results).  
      - The system displays a “Pending” status and informs the patient that results will be available shortly.  
   2. **Patient** selects a record that has no linked device data.  
      - The system still displays the record but shows a message “No device data linked.”  
   3. **Patient** chooses to export the record to PDF.  
      - The system generates a downloadable PDF and initiates the download.  
9. **Exceptions:**  
   1. **Authentication Failure** (occurs at step 2).  
      - The system denies access, displays an error message, and logs the attempt.  
   2. **Authorization Failure** (occurs at step 3).  
      - The system restricts access to the “My Medical Records” page, shows an “Access Denied” message, and logs the event.  
   3. **Record Not Found** (occurs at step 5).  
      - The system displays “Record not found” and offers to return to the list.  
   4. **Data Retrieval Error** (occurs at step 6 or 9).  
      - The system shows a generic error, attempts a retry, and if the retry fails, logs the incident and informs the patient that the data could not be loaded.  
   5. **Network Timeout** (occurs during any data fetch).  
      - The system displays a timeout message, offers a retry button, and logs the timeout.