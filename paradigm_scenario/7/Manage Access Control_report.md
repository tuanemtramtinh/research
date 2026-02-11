# Use Case Description Report: Manage Access Control

## 1. Use Case Name
**Manage Access Control**

## 2. Description
Enables an administrator to grant or revoke user access to specific modules or patient data.

## 3. Primary Actor
**Administrator**

## 4. Problem Domain Context
Develop an integrated patient‑care platform that:
1. Provides patients with secure mobile access to their full medical record, including measurements, test results, and prescription renewal.
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.
3. Allows doctors to view, edit, annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.
4. Permits nurses to edit and review records for clinical preparation.
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

## 5. Preconditions
- The administrator is authenticated and authorized to modify access control settings.  
- The target user’s account already exists in the system.  
- Relevant modules or patient data entries exist and are identifiable.

## 6. Postconditions
- The targeted user’s access permissions are updated (granted or revoked) as requested.  
- Audit logs record the change with timestamp, administrator identity, and affected resources.  
- The system notifies affected parties (e.g., via email or in‑app notification) of the permission change.

## 7. Main Flow
1. **Administrator** logs into the system and navigates to the *Access Control* management interface.  
2. The system displays a list of users and their current role assignments.  
3. **Administrator** selects a specific user to modify.  
4. The system presents the user’s current permissions across modules and patient records.  
5. **Administrator** chooses to *Grant* or *Revoke* access to a module, patient record, or specific data field.  
   - If granting, **Administrator** selects the desired scope (e.g., *View*, *Edit*, *Annotate*).  
   - If revoking, **Administrator** confirms the removal of permissions.  
6. **Administrator** reviews the proposed changes in a summary view.  
7. **Administrator** confirms the update.  
8. The system updates the user’s access rights in the database.  
9. The system logs the action, recording administrator ID, target user, changed permissions, and timestamp.  
10. The system sends a notification to the affected user (and optionally to other stakeholders).  
11. **Administrator** receives confirmation of successful completion and may continue managing other users or exit.

## 8. Alternative Flows
- **Alternative Flow A1 (Bulk Permission Update)**  
  1. *Administrator* selects multiple users from the list.  
  2. The system presents a bulk edit interface.  
  3. *Administrator* applies the same permission change to all selected users.  
  4. The system processes each change, logs individually, and provides a consolidated success/failure report.

- **Alternative Flow A2 (Role-Based Assignment)**  
  1. *Administrator* chooses to assign a predefined role (e.g., *Doctor*, *Nurse*, *Lab Technician*).  
  2. The system automatically applies the role’s default permissions to the user.  
  3. *Administrator* may fine‑tune individual permissions as needed.

## 9. Exceptions
1. **Exception E1 (Unauthorized Access Attempt)** – If an actor other than an administrator attempts to modify access controls, the system denies the request and logs the unauthorized attempt.  
2. **Exception E2 (Permission Conflict)** – When granting access that conflicts with existing system policies (e.g., granting a nurse edit rights on a restricted patient record), the system blocks the action, displays an error message, and suggests alternative permissions.  
3. **Exception E3 (Database Update Failure)** – If the database update fails (e.g., due to connectivity issues), the system rolls back any partial changes, notifies the administrator, and logs the error for audit.  
4. **Exception E4 (Invalid User Selection)** – If the selected user does not exist or has been deactivated, the system alerts the administrator and aborts the operation.  
5. **Exception E5 (Notification Failure)** – If sending the notification fails, the system logs the failure, retries automatically, and informs the administrator if the retry is unsuccessful.