## Use Case Name
**Schedule Appointment**

## Description
Handles appointment scheduling with automatic approvals, cancellations, time‑adjustments, and integration with room and staff calendars.

## Primary Actor
**Staff**

## Problem Domain Context
The system must provide a unified Electronic Medical Record Database that allows patients to securely view their own medical histories, test results and treatment details, while enabling authorized UHOPE staff, doctors, nurses, and receptionists to enter, edit, and search patient information only once and see only the records they are permitted to access; it must support appointment scheduling, including automatic approvals, cancellations, time‑adjustments, and integration with room and staff calendars; it must generate billing data automatically from appointment start/end times and treatment codes, and send invoices to patients within a specified window; it must offer a self‑service analytics portal for employees to query and analyze aggregated patient data; it must support mobile device integration for nurses to receive and upload measurements directly to the record; and it must provide secure authentication (unique ID/password plus optional two‑factor) and role‑based access control for all users.

## Preconditions
None.

## Postconditions
The appointment is recorded in the system, calendars are updated, billing data is generated, and any necessary notifications are sent.

## Main Flow
1. **Staff** logs into the system and navigates to the appointment scheduling module.  
2. **Staff** selects the patient from the patient lookup or creates a new patient record if not already present.  
3. **Staff** chooses a date and time for the appointment.  
4. The system checks room and staff availability, displaying conflicts if any.  
5. **Staff** confirms the selected slot.  
6. The system automatically approves the appointment if no conflicts exist; otherwise, it prompts **Staff** to resolve conflicts or reschedule.  
7. Upon approval, the system creates the appointment record, updates the room calendar, and updates the staff member’s calendar.  
8. The system generates billing data based on appointment duration and treatment codes.  
9. The system sends an appointment confirmation to the patient via email/SMS.  
10. The system logs the action for audit purposes.

## Alternative Flows
1a. *If a room or staff conflict is detected (step 4):*  
   - The system presents alternative time slots.  
   - **Staff** selects an alternative slot or cancels the scheduling.  

1b. *If the patient does not exist (step 2):*  
   - **Staff** creates a new patient profile and proceeds with steps 3–10.  

## Exceptions
1. **System** cannot retrieve room or staff availability (step 4).  
   - The system displays an error message and logs the issue for support.  

2. **System** fails to generate billing data due to missing treatment codes (step 8).  
   - The system prompts **Staff** to enter or correct treatment codes before proceeding.  

3. **System** fails to send confirmation to the patient (step 9).  
   - The system queues the notification for retry and logs the failure for manual follow‑up.  

4. **Authentication** fails at login (precondition).  
   - The system denies access and logs the attempt.