# Use Case Description Report

## 1. Use Case Name  
**View Medical History**

## 2. Description  
Allows patients to securely view their own medical histories, test results, and treatment details.

## 3. Primary Actor  
Patient

## 4. Problem Domain Context  
The system must provide a unified Electronic Medical Record Database that allows patients to securely view their own medical histories, test results and treatment details, while enabling authorized UHOPE staff, doctors, nurses, and receptionists to enter, edit, and search patient information only once and see only the records they are permitted to access; it must support appointment scheduling, including automatic approvals, cancellations, time‑adjustments, and integration with room and staff calendars; it must generate billing data automatically from appointment start/end times and treatment codes, and send invoices to patients within a specified window; it must offer a self‑service analytics portal for employees to query and analyze aggregated patient data; it must support mobile device integration for nurses to receive and upload measurements directly to the record; and it must provide secure authentication (unique ID/password plus optional two‑factor) and role‑based access control for all users.

## 5. Preconditions  
- The patient is registered in the system and has a valid account.  
- The patient has successfully authenticated (unique ID/password and optional two‑factor).  
- The patient’s medical records exist in the Electronic Medical Record Database.

## 6. Postconditions  
- The patient has viewed the requested medical history, test results, and treatment details.  
- No unauthorized data access occurs.  
- The system logs the view action for audit purposes.

## 7. Main Flow  
1. **Patient initiates action** – Patient logs into the portal and selects the “View Medical History” option.  
2. **System authenticates** – System verifies the patient’s credentials and two‑factor token if enabled.  
3. **System checks permissions** – System confirms the patient’s role and access rights to view personal records.  
4. **System retrieves records** – System queries the Electronic Medical Record Database for the patient’s history, test results, and treatment details.  
5. **System presents data** – System displays the records in a user‑friendly interface, allowing the patient to filter, sort, and download information.  
6. **Patient interacts with data** – Patient may drill down into specific entries, print reports, or export data.  
7. **System logs activity** – System records the view event, timestamp, and any interactions for audit and compliance.  
8. **Patient concludes session** – Patient logs out or ends the session.

## 8. Alternative Flows  
- **AF1 – Authentication Failure** (Step 2)  
  1. System prompts for re‑authentication or offers password reset.  
  2. If authentication fails three times, system locks the account and notifies security.  

- **AF2 – No Records Found** (Step 4)  
  1. System displays “No medical history available” message.  
  2. Patient is offered to contact support or schedule a new appointment.  

- **AF3 – Data Export Requested** (Step 6)  
  1. System verifies export permissions.  
  2. System generates secure PDF/CSV file and prompts download or email.  

## 9. Exceptions  
- **E1 – Database Unavailable** (Step 4)  
  1. System displays error message “Unable to retrieve records at this time.”  
  2. System logs the incident and queues a retry.  

- **E2 – Permission Denied** (Step 3)  
  1. System shows “Access denied” message.  
  2. System logs the attempt and advises contacting support.  

- **E3 – Session Timeout** (Any step)  
  1. System terminates the session and redirects to login page.  
  2. System records the timeout event for audit.  

- **E4 – Two‑Factor Failure** (Step 2)  
  1. System prompts for alternative second factor or offers recovery options.  
  2. After three failures, account is temporarily locked.