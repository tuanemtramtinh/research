## Use Case Name
**View Medical Record**

## Description
Allows a patient to securely access their full medical record, including measurements, test results, and prescription history via a mobile app.

## Primary Actor
Patient

## Problem Domain Context
Develop an integrated patient‑care platform that:

1. Provides patients with secure mobile access to their full medical record, including measurement, test results and prescription renewal.  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.  
3. Allows doctors to view, edit and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.  
4. Permits nurses to edit and review records for clinical preparation.  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

## Preconditions
- Patient has a registered account in the mobile app.  
- Patient has completed two‑step authentication (if enabled).  
- Patient has at least one medical record entry in the system.  
- Network connectivity is available.

## Postconditions
- The patient’s mobile device displays the requested medical record.  
- An audit log entry is created noting the record access.  
- The system updates the last‑access timestamp for the patient’s record.

## Main Flow
1. **Patient** opens the mobile app and navigates to the *Medical Records* section.  
2. The app **verifies** the patient’s current authentication status.  
3. The app **requests** the patient’s consent to view the requested record (if consent is not already granted).  
4. The patient **provides** consent via a confirmation dialog.  
5. The app **queries** the backend for the patient’s latest medical record data.  
6. The backend **retrieves** the record from the database, applying any role‑based and consent filters.  
7. The backend **returns** the record data to the app.  
8. The app **renders** the record, displaying measurements, test results, prescription history, and any attached documents.  
9. The patient **confirms** that the displayed information is complete.  
10. The app **logs** the access event with timestamp and device identifier.  
11. The patient’s session **continues** until they exit the record view.

## Alternative Flows
1.1 **Consent Not Granted**  
&nbsp;&nbsp;• The app prompts the patient to grant consent.  
&nbsp;&nbsp;• If the patient declines, the use case terminates with a message “Access denied.”

1.2 **Authentication Expired**  
&nbsp;&nbsp;• The app detects an expired session.  
&nbsp;&nbsp;• The patient is redirected to the login screen.  
&nbsp;&nbsp;• After successful re‑authentication, the app resumes from step 5.

1.3 **Record Not Found**  
&nbsp;&nbsp;• The backend returns a “record not found” error.  
&nbsp;&nbsp;• The app displays “No medical record available.” and offers to contact support.

## Exceptions
1. **Network Failure (Step 5)**  
&nbsp;&nbsp;• The app shows “Network error. Please retry.” and allows the patient to retry the request.

2. **Data Integrity Error (Step 6)**  
&nbsp;&nbsp;• The backend logs the inconsistency and returns an error.  
&nbsp;&nbsp;• The app informs the patient that the record could not be retrieved and suggests contacting support.

3. **Unauthorized Access (Step 6)**  
&nbsp;&nbsp;• The backend detects a role or consent violation.  
&nbsp;&nbsp;• The app displays “Access denied due to insufficient permissions.” and logs the attempt.

4. **Audit Log Failure (Step 10)**  
&nbsp;&nbsp;• The audit logging subsystem fails.  
&nbsp;&nbsp;• The app proceeds but records the failure in a local buffer to be sent when connectivity is restored.