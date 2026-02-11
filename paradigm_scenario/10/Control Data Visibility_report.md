## Use Case: Control Data Visibility

| Item | Details |
|------|---------|
| **Use Case Name** | Control Data Visibility |
| **Description** | Administrator controls data visibility across departments |
| **Primary Actor** | Administrator |
| **Problem Domain Context** | The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments. |
| **Preconditions** | None. |
| **Postconditions** | The system enforces the updated visibility rules, and a log entry records the change. |

### Main Flow
1. **Administrator logs in** using DigiD or standard credentials.  
2. **System authenticates** the administrator and displays the *Data Visibility Management* dashboard.  
3. **Administrator selects a department** (e.g., Cardiology, Oncology).  
4. **System presents current visibility settings** for that department, including who can view which data categories (patient records, lab results, device data, etc.).  
5. **Administrator chooses to modify visibility** for a specific data category (e.g., restrict lab results to specialists only).  
6. **System prompts for confirmation** and optional justification text.  
7. **Administrator confirms** the change.  
8. **System applies the new visibility rule**, updating access control lists (ACLs) and re‑evaluating all affected user sessions.  
9. **System logs** the change with timestamp, administrator ID, and description.  
10. **System displays a success message** and updates the dashboard to reflect the new settings.

### Alternative Flows
- **AF1 (Change applies to multiple departments)**  
  3a. Administrator selects *Apply to all departments*.  
  4a. System shows aggregated visibility settings.  
  5a. Administrator modifies settings as in step 5.  
  6a. System confirms bulk operation.  
  8a. System updates ACLs for all selected departments.  
  9a. System logs bulk change.  

- **AF2 (Administrator wants to revert a previous change)**  
  3b. Administrator selects *View change history*.  
  4b. System lists recent visibility changes with timestamps.  
  5b. Administrator selects a change to revert.  
  6b. System prompts for confirmation.  
  7b. Administrator confirms revert.  
  8b. System restores previous ACLs.  
  9b. System logs the revert action.  

- **AF3 (Administrator denies permission to edit visibility)**  
  2a. System detects insufficient rights and displays *Access Denied* message.  
  3a. Administrator logs out or requests higher privileges.

### Exceptions
- **E1 (Authentication Failure)** – Step 1: If credentials are invalid, system displays *Login Failed* and denies access.  
- **E2 (Insufficient Permissions)** – Step 3: If the administrator lacks rights to modify visibility, system shows *You do not have permission to modify visibility settings* and returns to the dashboard.  
- **E3 (Concurrent Modification Conflict)** – Step 8: If another administrator modifies the same visibility rule simultaneously, system detects conflict, prompts the current administrator to refresh and reconcile changes.  
- **E4 (System Error during ACL Update)** – Step 8: If the ACL update fails (e.g., database error), system rolls back to previous state, logs the error, and displays *An error occurred while applying changes. Please try again later*.  
- **E5 (Logging Failure)** – Step 9: If the audit log cannot record the change, system still applies visibility rules but displays a warning *Audit log update failed; please contact support*.