**Use Case Name:** Upload Measurements  
**Description:** Allows nurses to receive and upload measurements directly to the electronic record via mobile device integration.  
**Primary Actor:** Nurse  
**Problem Domain Context:**  
The system must provide a unified Electronic Medical Record Database that allows patients to securely view their own medical histories, test results and treatment details, while enabling authorized UHOPE staff, doctors, nurses, and receptionists to enter, edit, and search patient information only once and see only the records they are permitted to access. It must support appointment scheduling, including automatic approvals, cancellations, time‑adjustments, and integration with room and staff calendars; generate billing data automatically from appointment start/end times and treatment codes, and send invoices to patients within a specified window; offer a self‑service analytics portal for employees to query and analyze aggregated patient data; support mobile device integration for nurses to receive and upload measurements directly to the record; and provide secure authentication (unique ID/password plus optional two‑factor) and role‑based access control for all users.  

**Preconditions:**  
- The nurse is authenticated and authorized with the “Nurse” role.  
- The nurse’s mobile device is registered and connected to the UHOPE mobile app.  
- The patient has an active appointment scheduled and the measurement device is paired with the nurse’s device.  

**Postconditions:**  
- The measurement data is stored in the patient’s electronic record with a timestamp.  
- The record is flagged as “Measurement Uploaded” and visible to authorized staff.  
- If applicable, billing data is updated to reflect the measurement event.  

**Main Flow:**  
1. **Nurse** opens the UHOPE mobile app and selects the “Upload Measurements” function.  
2. The app retrieves the list of upcoming appointments for the day.  
3. **Nurse** selects the appointment for which measurements will be uploaded.  
4. The app initiates a connection to the paired measurement device (e.g., blood pressure cuff).  
5. The device transmits the measurement data to the mobile app.  
6. The app validates the data format and checks for any missing mandatory fields.  
7. The app displays a confirmation screen showing the captured values and the patient’s identifier.  
8. **Nurse** reviews the data and confirms the upload.  
9. The app sends the measurement data to the backend service via a secure API.  
10. The backend service records the data in the patient’s electronic record, timestamps it, and logs the nurse’s user ID.  
11. The backend triggers any necessary billing updates and analytics flags.  
12. The app displays a success message and updates the local appointment status to “Measurements Uploaded.”  

**Alternative Flows:**  
- **3a.** If the selected appointment has already had measurements uploaded, the app prompts the nurse to confirm overwriting or aborting.  
- **4a.** If the device is not reachable, the app offers to reconnect or select a different device.  
- **6a.** If data validation fails (e.g., missing units), the app highlights the issue and allows the nurse to correct or cancel.  
- **8a.** If the nurse chooses to cancel, the app aborts the upload and returns to the appointment list.  

**Exceptions:**  
- **5a.** *Device communication error*: The app logs the error, informs the nurse, and retries up to three times before aborting.  
- **6b.** *Invalid data format*: The app displays an error detailing the missing or malformed fields and halts the upload.  
- **9a.** *API failure*: The app shows a network error, offers to retry, and queues the data for retry on next connectivity.  
- **10a.** *Database write error*: The backend logs the failure, returns a failure response, and the app displays an error message allowing the nurse to retry after investigation.  
- **12a.** *Billing update failure*: The backend logs the issue; the app notifies the nurse that the measurement was uploaded but billing may be delayed, and offers to contact support.