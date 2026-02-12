# Use Case Description Report

## 1. Use Case Name
**Schedule Appointment**

## 2. Description
Enables a doctor to schedule appointments while considering lab and pharmacy availability.

## 3. Primary Actor
Doctor

## 4. Problem Domain Context
Develop an integrated patient‑care platform that:
1. Provides patients with secure mobile access to their full medical record, including measurements, test results, and prescription renewal.  
2. Enables patients to share wearable‑device data and selectively grant or revoke access to healthcare providers.  
3. Allows doctors to view, edit, and annotate patient records, schedule appointments, request and track laboratory tests, and issue prescriptions while managing pharmacy inventory visibility.  
4. Permits nurses to edit and review records for clinical preparation.  
5. Supports receptionists with patient identification, appointment scheduling, and room assignment using doctor and lab availability.  
6. Equips laboratory technicians with a dashboard of pending tests, urgency, and requester details, along with the ability to input results and attach notes.  
7. Offers pharmacists the ability to verify patient ID, dosage, and prescription details.  
8. Gives system administrators the capability to configure user roles, enforce two‑step authentication, and adjust access rights across the system.

## 5. Preconditions
- Doctor is authenticated with two‑step authentication.  
- Doctor’s calendar and lab/pharmacy availability data are up‑to‑date.  
- Patient record is accessible and current.  

## 6. Postconditions
- Appointment is created in the system’s scheduling module.  
- Doctor’s calendar reflects the new appointment.  
- Lab and pharmacy schedules are updated to reflect the appointment’s requirements.  
- Confirmation notification is sent to the patient and relevant staff.

## 7. Main Flow
1. **Doctor** logs into the platform and navigates to the scheduling module.  
2. **Doctor** selects a patient from the patient list.  
3. **Doctor** chooses a date and time slot from the available calendar view.  
4. System **checks** lab and pharmacy availability for the selected slot.  
5. If **available**, system **proposes** the slot to the doctor.  
6. **Doctor** confirms the appointment.  
7. System **creates** the appointment record and updates the doctor’s calendar.  
8. System **reserves** the necessary lab and pharmacy resources.  
9. System **sends** confirmation to the patient and relevant staff (receptionist, nurse, lab tech, pharmacist).  
10. Appointment is **logged** in the patient’s medical record for future reference.

## 8. Alternative Flows
1. **Lab/Pharmacy Unavailable** (branches from step 4):  
   1.1 System **alerts** the doctor that the selected slot conflicts with lab or pharmacy availability.  
   1.2 Doctor **chooses** an alternative slot.  
   1.3 Flow returns to step 4.  

2. **Patient Declines Appointment** (branches from step 9):  
   2.1 Patient **rejects** the appointment via notification.  
   2.2 System **marks** the appointment as declined and frees the resources.  
   2.3 Doctor **re‑schedules** or **cancels** the appointment.  

## 9. Exceptions
1. **Authentication Failure** (occurs before step 1):  
   1.1 System **rejects** access and prompts for re‑authentication.  

2. **Calendar Sync Error** (occurs during step 4):  
   2.1 System **logs** the error and displays a message: “Unable to verify availability. Please try again later.”  
   2.2 Doctor **may** retry after a short delay.  

3. **Resource Reservation Failure** (occurs during step 8):  
   3.1 System **rolls back** any provisional changes.  
   3.2 System **notifies** the doctor that the appointment could not be scheduled due to resource constraints.  
   3.3 Doctor **selects** a different slot or cancels the attempt.  

4. **Notification Delivery Failure** (occurs during step 9):  
   4.1 System **logs** the failure and retries up to 3 times.  
   4.2 If still unsuccessful, system **alerts** the receptionist to manually confirm the appointment with the patient.