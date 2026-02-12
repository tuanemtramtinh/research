# Use Case Description Report

## 1. Use Case Name
**Manage Device Authorizations**

## 2. Description
Administrator manages authorizations for external devices.

## 3. Primary Actor
**Administrator**

## 4. Problem Domain Context
The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.

## 5. Preconditions
- Administrator is authenticated via DigiD or standard credentials.  
- Administrator has access to the Device Authorization Management module.  
- The system contains a list of external devices registered within the network.

## 6. Postconditions
- The device authorization status is updated in the system.  
- Updated permissions are reflected in the device‑to‑user access matrix.  
- Audit logs record the change with timestamp and administrator identity.  
- A notification is optionally sent to affected users and device owners.

## 7. Main Flow
1. **Administrator logs in** via DigiD or standard credentials.  
2. **Administrator navigates** to the Device Authorization Management module.  
3. **System displays** a list of all registered external devices with current authorization status.  
4. **Administrator selects** a device from the list to manage authorizations.  
5. **System shows** device details and a list of users/groups currently authorized.  
6. **Administrator chooses** to add, remove, or modify user/group authorizations.  
   - 6a. **Add**: selects a user/group, assigns permissions (read/write, data types), and confirms.  
   - 6b. **Remove**: selects a user/group, confirms removal.  
   - 6c. **Modify**: changes permission levels or data types for an existing user/group.  
7. **System validates** the changes against policy rules (e.g., no duplicate permissions, minimum required roles).  
8. **System updates** the authorization database with the new settings.  
9. **System logs** the action with timestamp, administrator ID, and details of changes.  
10. **System notifies** affected users and device owners of the authorization update (if configured).  
11. **Administrator reviews** the updated list to confirm changes.  
12. **Administrator logs out** or returns to the main dashboard.

## 8. Alternative Flows
- **AF1 – Device Not Found (Step 4)**  
  1. Administrator selects a device that no longer exists.  
  2. System displays an error message: "Device not found."  
  3. Administrator is prompted to return to the device list or search again.  

- **AF2 – Permission Conflict (Step 7)**  
  1. Administrator attempts to assign permissions that conflict with existing policies (e.g., read-only user granted write access).  
  2. System rejects the change and displays a conflict explanation.  
  3. Administrator must resolve the conflict before proceeding.  

- **AF3 – Notification Failure (Step 10)**  
  1. System fails to send notification due to network issue.  
  2. System logs the failure and flags the notification for retry.  
  3. Administrator receives a system alert regarding the pending notification.

## 9. Exceptions
- **E1 – Authentication Failure (Step 1)**  
  1. System detects invalid credentials.  
  2. Displays error message and prompts retry.  

- **E2 – Authorization Error (Steps 4–7)**  
  1. Administrator lacks required system role to modify device authorizations.  
  2. System denies access and logs the attempt.  

- **E3 – Database Error (Step 8)**  
  1. System cannot commit changes due to database connectivity or transaction failure.  
  2. System rolls back changes, displays error, and logs the incident.  

- **E4 – Audit Log Failure (Step 9)**  
  1. System fails to record the audit entry.  
  2. System alerts administrators and stores the log entry for later reconciliation.  

- **E5 – Notification Service Down (Step 10)**  
  1. System cannot send emails/SMS.  
  2. System queues the notification for later delivery and logs the failure.