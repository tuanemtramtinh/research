## Use Case: Manage user access

| Item | Details |
|------|---------|
| **Use Case Name** | Manage user access |
| **Description** | System administrators centrally store all data, enforce GDPR controls, and manage user access levels. |
| **Primary Actor** | System Administrator |
| **Problem Domain Context** | The system must:<br>1. Enable doctors to create, view, and electronically transfer prescriptions to pharmacies.<br>2. Allow patients to view, edit personal information, and request new prescriptions through online or mobile appointment scheduling.<br>3. Provide nurses, receptionists, and doctors with secure access to patients’ electronic medical records, including medical history, personal details, and notes.<br>4. Support mobile device synchronization for doctors and patients to maintain consistent workflow across platforms.<br>5. Allow pharmacists to view transferred prescriptions and manage medication dispensing.<br>6. Grant system administrators the ability to centrally store all data, integrate electronic medical records, enforce GDPR and privacy controls, and manage user access levels.<br>7. Supply systems analysts with anonymized clinical data, bed occupancy, room scheduling, and employee schedules for analytics.<br>8. Record appointment start and end times for duration calculations.<br>9. Support migration of data from legacy systems to the new platform.<br>10. Implement secure authentication and authorization mechanisms to restrict system component access to authorized users. |
| **Preconditions** | None. |
| **Postconditions** | User access levels are updated, audit logs recorded, and GDPR compliance status refreshed. |

### Main Flow
1. **Administrator logs in** to the admin console using secure authentication.  
2. Administrator **navigates to the User Management module**.  
3. Administrator **views the list of users** and their current roles/permissions.  
4. Administrator **selects a user** to modify or creates a new user.  
5. Administrator **edits the user’s role** (e.g., Doctor, Nurse, Pharmacist, Patient, Analyst, or Admin).  
6. Administrator **configures granular permissions** (e.g., read/write access to specific data sets).  
7. Administrator **sets GDPR flags** (e.g., data retention period, consent status).  
8. Administrator **saves changes** and triggers an audit log entry.  
9. System **validates the new configuration** against policy constraints.  
10. System **notifies the user** (via email/SMS) of role changes or new account creation.  
11. Administrator **reviews audit logs** to confirm successful update.  
12. Use case **terminates** with the system reflecting updated access controls.

### Alternative Flows
1. **User not found** (alternative to step 4): Administrator selects *Create new user* instead.  
2. **Role conflicts** (alternative to step 7): System prompts administrator to resolve policy conflicts before saving.  
3. **GDPR compliance warning** (alternative to step 7): System displays a warning if the role requires special GDPR handling (e.g., patient data access) and requires administrator confirmation.

### Exceptions
1. **Authentication failure** (step 1): System denies access and logs the attempt.  
2. **Authorization failure** (step 1‑2): System displays “Insufficient privileges” and aborts the use case.  
3. **Validation error** (step 9): System highlights invalid fields, prompts correction, and prevents saving until resolved.  
4. **Audit log failure** (step 8): System retries once; if still failing, displays error and logs incident for admin review.  
5. **Notification failure** (step 10): System logs the failure and queues a retry; administrator is notified of the issue.