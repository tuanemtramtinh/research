## Use Case Name
Authenticate User

## Description
Provides secure authentication using unique ID/password with optional two‑factor and enforces role‑based access control for all users.

## Primary Actor
Patient

## Problem Domain Context
The system must provide a unified Electronic Medical Record Database that allows patients to securely view their own medical histories, test results and treatment details, while enabling authorized UHOPE staff, doctors, nurses, and receptionists to enter, edit, and search patient information only once and see only the records they are permitted to access; it must support appointment scheduling, including automatic approvals, cancellations, time‑adjustments, and integration with room and staff calendars; it must generate billing data automatically from appointment start/end times and treatment codes, and send invoices to patients within a specified window; it must offer a self‑service analytics portal for employees to query and analyze aggregated patient data; it must support mobile device integration for nurses to receive and upload measurements directly to the record; and it must provide secure authentication (unique ID/password plus optional two‑factor) and role‑based access control for all users.

## Preconditions
None.

## Postconditions
* User is authenticated and assigned a session token.
* User’s role is loaded into the session context.
* Permissions for the user are enforced for subsequent interactions.

## Main Flow
1. **Patient** initiates the login process by navigating to the login screen.  
2. System presents fields for **Patient** to enter **User ID** and **Password**.  
3. **Patient** submits credentials.  
4. System validates **User ID** and **Password** against the authentication database.  
5. If credentials are valid, system checks whether two‑factor authentication (2FA) is enabled for the user.  
6. **If 2FA is enabled**:  
   6.1. System generates a one‑time code (OTC) and sends it to the user’s registered device or email.  
   6.2. System prompts the user to enter the OTC.  
   6.3. **Patient** submits the OTC.  
   6.4. System verifies the OTC.  
7. Upon successful verification (or if 2FA is not enabled), system creates a session token and assigns the user’s role.  
8. System redirects the user to the appropriate dashboard based on role privileges.

## Alternative Flows
1. **Incorrect Credentials (Branch from Step 4)**  
   1.1. System displays an error message: "Invalid User ID or Password."  
   1.2. **Patient** is prompted to retry login.  
2. **2FA Not Received (Branch from Step 6.1)**  
   2.1. System offers the option to resend the OTC.  
   2.2. **Patient** requests resend; system re‑sends OTC.  
3. **2FA Code Expired (Branch from Step 6.3)**  
   3.1. System notifies: "The code has expired."  
   3.2. System allows the user to request a new code.  
4. **Account Lockout After Multiple Failed Attempts (Branch from Step 4)**  
   4.1. System locks the account temporarily.  
   4.2. System displays: "Account locked due to multiple failed attempts. Please try again after X minutes."  

## Exceptions
1. **Database Unavailable (Occurs at Step 4)**  
   1.1. System displays: "Authentication service is temporarily unavailable. Please try again later."  
   1.2. Login process terminates.  
2. **Missing 2FA Device/Email (Occurs at Step 5)**  
   2.1. System prompts: "No 2FA device registered. Would you like to set up 2FA?"  
   2.2. If user declines, system proceeds without 2FA.  
3. **Session Token Generation Failure (Occurs at Step 7)**  
   3.1. System logs the error.  
   3.2. System displays: "Unable to establish a session. Contact support."  
   3.3. Login process terminates.