# Use Case Description Report

## 1. Use Case Name
**Order And Rate Lab Tests**

## 2. Description
Doctor orders and rates lab tests for patients.

## 3. Primary Actor
**Doctor**

## 4. Problem Domain Context
The system must provide secure, always‑available access for all users, enabling patients to view, edit, and manage their own medical records and linked external device data; doctors to restrict their view to assigned patient records, add, edit, and annotate records, link multiple specialists, and order and rate lab tests; lab specialists to upload results, link tests to records, and annotate results; nurses and receptionists to link external devices, record patient data, and manage appointments, beds, and doctor assignments; and administrators to authenticate via DigiD or standard credentials, manage device authorizations, and control data visibility across departments.

## 5. Preconditions
- The doctor is authenticated and authorized to access the patient’s record.  
- The patient’s record is accessible and currently open in the system.  
- The lab test catalogue is populated and up‑to‑date.  
- The doctor has sufficient privileges to order and rate lab tests.

## 6. Postconditions
- The lab test is added to the patient’s record with an order status of **Pending**.  
- A notification is sent to the lab specialist to perform the test.  
- The doctor’s rating (if provided) is stored and associated with the test.  
- The patient’s record reflects the new lab test entry.

## 7. Main Flow
1. **Doctor** opens the patient’s record.  
2. **Doctor** selects the “Order Lab Test” function.  
3. The system displays the lab test catalogue.  
4. **Doctor** searches for and selects the desired test.  
5. The system prompts the doctor to confirm the order and optionally add a rating.  
6. **Doctor** confirms the order and enters a rating (e.g., urgency, priority).  
7. The system creates a new lab test entry, sets status to **Pending**, and links it to the patient’s record.  
8. The system triggers an alert to the lab specialist and logs the order.  
9. **Doctor** receives confirmation of successful order placement.

## 8. Alternative Flows
- **A1 – Rating Not Provided**  
  - *Trigger:* Step 6  
  - *Action:* The doctor skips the rating field.  
  - *Result:* The test is ordered with a default rating (e.g., **Standard**).  
- **A2 – Test Already Ordered**  
  - *Trigger:* Step 4  
  - *Action:* System checks for existing pending orders of the same test.  
  - *Result:* System displays a warning and offers to duplicate or cancel the existing order.  

## 9. Exceptions
- **E1 – Unauthorized Access**  
  - *Trigger:* Step 1 or 2  
  - *Handling:* System denies access, logs the attempt, and displays an error message.  
- **E2 – Catalogue Unavailable**  
  - *Trigger:* Step 3  
  - *Handling:* System shows an error, offers to retry or use a local offline catalogue.  
- **E3 – Duplicate Order Conflict**  
  - *Trigger:* Step 4 (alternative flow A2)  
  - *Handling:* System prevents duplicate orders, prompts the doctor to modify or cancel the existing one.  
- **E4 – System Failure During Order Creation**  
  - *Trigger:* Step 7  
  - *Handling:* Transaction is rolled back, an error is logged, and the doctor is notified to retry.