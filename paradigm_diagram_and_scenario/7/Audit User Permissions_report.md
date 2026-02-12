# Use Case Description Report

**Use Case Name:**  
Audit User Permissions  

**Description:**  
Allows an administrator to review and modify access permissions for all users in the system.  

**Primary Actor:**  
Administrator  

**Problem Domain Context:**  
Develop an integrated patient‑care platform that  
1. Provides patients with secure mobile access to their full medical record, including measurement, test results and prescription renewal;  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers;  
3. Allows doctors to view, edit and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility;  
4. Permits nurses to edit and review records for clinical preparation;  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability;  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes;  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details;  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.  

**Preconditions:**  
- Administrator is authenticated and has the “Admin” role.  
- The system has at least one user with a defined role and permission set.  

**Postconditions:**  
- The permission matrix for all users is updated as per the administrator’s changes.  
- Audit logs record the changes with timestamps and administrator identity.  

**Main Flow:**  
1. **Administrator logs into the system** using credentials and completes any required two‑step authentication.  
2. **Administrator navigates to the “User Management” module** from the main dashboard.  
3. **System displays a list of all users** along with their current roles and permissions.  
4. **Administrator selects a user** whose permissions need review or modification.  
5. **System presents the user’s permission details** in a detailed view or editable form.  
6. **Administrator edits the permissions** by selecting/deselecting checkboxes, changing roles, or adjusting specific access rights.  
7. **Administrator saves the changes** by clicking the “Update Permissions” button.  
8. **System validates the changes** (e.g., checks for prohibited privilege escalation).  
9. **System updates the permission records** in the database and logs the action.  
10. **System confirms success** to the administrator with a notification and updates the user list view.  

**Alternative Flows:**  
1.1. *User Not Found* – If the selected user does not exist, the system displays an error message and returns to step 3.  
2.1. *Insufficient Privileges* – If the administrator lacks the “Permission Management” sub‑role, the system denies access at step 3 and prompts for additional authorization.  

**Exceptions:**  
1.1. *Authentication Failure* – If login fails, the system displays an error and halts the flow.  
4.1. *Concurrent Modification* – If another administrator modifies the same user’s permissions simultaneously, the system detects a conflict at step 8 and prompts the administrator to reload or merge changes.  
6.1. *Invalid Permission Configuration* – If the administrator attempts to assign permissions that violate system rules (e.g., giving a nurse access to prescription issuance), the system rejects the update at step 8 and highlights the conflict.  

---