# Use Case Description Report

## 1. Use Case Name  
**Authenticate Via Digi D**

## 2. Description  
Administrator authenticates users via DigiD or standard credentials.

## 3. Primary Actor  
Administrator

## 4. Problem Domain Context  
The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.

## 5. Preconditions  
- Administrator is logged into the administrative console.  
- DigiD service is reachable and operational.  
- User database contains the user’s stored credentials (if using standard authentication).  

## 6. Postconditions  
- User is authenticated and granted access according to their role permissions.  
- An authentication log entry is created.  
- Session token is issued to the user.  

## 7. Main Flow  
1. **Administrator initiates login** – Administrator selects “Authenticate Via DigiD” or “Standard Credentials” on the login screen.  
2. **System presents authentication options** – System displays DigiD OAuth flow or standard username/password fields.  
3. **Administrator selects DigiD** – If DigiD is chosen, system redirects to the DigiD login portal.  
4. **Administrator enters DigiD credentials** – Administrator provides DigiD username and password to the DigiD portal.  
5. **DigiD validates credentials** – DigiD service verifies credentials and returns an authentication token.  
6. **System receives DigiD token** – System receives the token and verifies its signature and expiry.  
7. **System queries user profile** – System uses the token to retrieve the user’s profile (role, permissions, etc.) from the DigiD API.  
8. **System creates local user session** – System generates a local session token, maps DigiD user to system user, and stores session data.  
9. **System logs the authentication event** – System records the timestamp, user ID, authentication method, and outcome.  
10. **System redirects to user dashboard** – User is taken to the appropriate dashboard based on role.  

If the administrator chooses **Standard Credentials**:  
- Steps 3–4 are replaced by the administrator entering a username and password into the local login form.  
- Steps 5–6 involve local credential verification against the user database.  
- Steps 7–10 follow similarly, using the local user profile.

## 8. Alternative Flows  
**A1 – DigiD service unavailable**  
- **Step 3** – System detects DigiD service unavailability.  
- System displays an error message: “DigiD authentication service is currently unavailable. Please try again later or use standard credentials.”  
- Administrator may choose to use Standard Credentials (return to main flow step 3).  

**A2 – Administrator cancels DigiD authentication**  
- **Step 4** – Administrator clicks “Cancel” in the DigiD portal.  
- System aborts the DigiD flow, clears any temporary tokens, and returns to the login screen.  

**A3 – Standard credentials incorrect**  
- **Step 5** – System fails to match username/password.  
- System displays “Invalid username or password.”  
- Administrator is prompted to retry (loop back to step 5).  

## 9. Exceptions  
1. **Invalid DigiD token** – If the token received from DigiD fails validation (signature or expiry), the system logs the failure, displays “Authentication failed. Please try again.”, and aborts the flow.  
2. **User not found in system** – If the DigiD profile does not map to an existing system user, the system logs the event, displays “User not registered in the system.”, and offers registration options.  
3. **Database connectivity error** – If the system cannot reach the user database during credential verification, the system logs the error, displays “System error. Please contact support.”, and halts the authentication process.  
4. **Session token generation failure** – If session creation fails, the system logs the issue, displays “Unable to create session. Please try again.”, and aborts the flow.