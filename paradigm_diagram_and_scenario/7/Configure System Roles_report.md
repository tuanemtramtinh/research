# Use Case: Configure System Roles

## 1. Use Case Name
**Configure System Roles**

## 2. Description
Provides system administrators the ability to configure user roles, enforce two‑step authentication, and adjust access rights across the platform.

## 3. Primary Actor
**Administrator**

## 4. Problem Domain Context
Develop an integrated patient‑care platform that:

1. Provides patients with secure mobile access to their full medical record, including measurements, test results, and prescription renewal.  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.  
3. Allows doctors to view, edit and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.  
4. Permits nurses to edit and review records for clinical preparation.  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

## 5. Preconditions
- The administrator is authenticated and logged into the system.  
- The administrator has the necessary managerial privileges to modify role settings.  
- The system has at least one predefined role schema available.

## 6. Postconditions
- Updated role configuration is persisted in the system database.  
- Two‑step authentication settings are applied to the relevant user accounts.  
- Access rights are refreshed across all modules to reflect the new role definitions.  
- Audit log records the changes made by the administrator.

## 7. Main Flow
1. **Administrator** logs into the system and navigates to the *Role Management* dashboard.  
2. The system displays a list of existing roles and their current permissions.  
3. **Administrator** selects *Create New Role* or *Edit Existing Role*.  
4. The system presents a role configuration form (role name, description, base permissions).  
5. **Administrator** defines or modifies permissions (e.g., read/write access to patient records, lab result entry, prescription issuance).  
6. **Administrator** selects the *Two‑Step Authentication* toggle to enforce or disable for the role.  
7. **Administrator** reviews the configuration summary and clicks *Save*.  
8. The system validates the input, persists the changes, and updates the role database.  
9. The system propagates updated permissions to all affected user sessions and logs the action.  
10. The system displays a success confirmation to the **Administrator**.

## 8. Alternative Flows
1.1. *Creating a Role with Inherited Permissions*  
&nbsp;&nbsp;• The administrator chooses to inherit permissions from an existing role.  
&nbsp;&nbsp;• The system copies the base permissions and allows the administrator to add or remove specific rights.  

2.1. *Editing Permissions for an Existing Role*  
&nbsp;&nbsp;• The administrator modifies only a subset of permissions.  
&nbsp;&nbsp;• The system updates the difference set without overwriting unchanged rights.  

3.1. *Role Deletion*  
&nbsp;&nbsp;• The administrator selects *Delete Role* for a role that has no assigned users.  
&nbsp;&nbsp;• The system confirms deletion, removes the role, and logs the action.

## 9. Exceptions
1.1. **Invalid Input**  
&nbsp;&nbsp;• If required fields (e.g., role name) are missing or contain invalid characters, the system displays an error message and prevents saving.  

2.1. **Duplicate Role Name**  
&nbsp;&nbsp;• If the role name already exists, the system alerts the administrator and requests a unique name.  

3.1. **Permission Conflict**  
&nbsp;&nbsp;• If the administrator attempts to grant a permission that conflicts with system-wide policies (e.g., granting prescription issuance to a non‑physician role), the system rejects the change and provides an explanation.  

4.1. **Concurrency Issue**  
&nbsp;&nbsp;• If another administrator modifies the same role simultaneously, the system detects the conflict and prompts the administrator to refresh or merge changes.  

5.1. **Database Failure**  
&nbsp;&nbsp;• If the system cannot persist changes due to a database error, it rolls back the transaction, displays an error message, and logs the failure for troubleshooting.