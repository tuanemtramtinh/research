# Use Case Description Report

## 1. Use Case Name
**Restrict View To Assigned Patients**

## 2. Description
Doctor restricts view to only assigned patient records.

## 3. Primary Actor
Doctor

## 4. Problem Domain Context
The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.

## 5. Preconditions
- Doctor is authenticated and logged into the system.  
- Doctor has one or more patients assigned to them.  
- System has up-to-date patient assignment data.

## 6. Postconditions
- The doctor’s view is limited to the records of patients assigned to them.  
- All other patient records remain inaccessible to the doctor.  
- System logs the visibility restriction action for audit purposes.

---

## 7. Main Flow
1. **Doctor** logs into the system.  
2. System authenticates the doctor and retrieves their profile.  
3. **Doctor** navigates to the “Patient Management” dashboard.  
4. System displays a list of all patients.  
5. **Doctor** selects the “Restrict View” option.  
6. System prompts the doctor to confirm the restriction action.  
7. **Doctor** confirms the restriction.  
8. System updates the doctor’s visibility settings in the database, marking all non‑assigned patient records as hidden.  
9. System refreshes the dashboard to show only assigned patient records.  
10. System records the action in the audit trail with timestamp and doctor ID.  

---

## 8. Alternative Flows
1. **(Flow 5.1)**  
   - If the doctor selects “Cancel” instead of confirming at step 6, the system aborts the restriction process and returns to the dashboard without changing visibility.  

2. **(Flow 5.2)**  
   - If the doctor already has a restriction in place and attempts to apply it again, the system displays a message: “Restriction already active.” and offers to remove the restriction instead.  

---

## 9. Exceptions
1. **(Step 2)**  
   - *Authentication Failure*: If the system fails to authenticate the doctor (e.g., wrong credentials, account locked), it displays an error message and terminates the session.  

2. **(Step 8)**  
   - *Database Update Failure*: If the system cannot update visibility settings due to a database error, it rolls back any partial changes, displays an error message, and logs the incident for support.  

3. **(Step 10)**  
   - *Audit Trail Failure*: If recording the action fails, the system still applies the restriction but flags the incident for later manual audit and notifies the system administrator.