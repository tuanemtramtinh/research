## Use Case Name
**View live updates**

## Description
Fan receives real‑time notifications of match events, schedule changes, and budget info in local language and currency.

## Primary Actor
Fan

## Problem Domain Context
The IT system must enable the IFA to:

* Define, change, and enforce league and budget policies.
* Automatically generate match schedules and assign referees while respecting stakeholder constraints.
* Record and display real‑time game event data.
* Provide real‑time budget and resource monitoring for teams.
* Support team management of finances, social media pages, and content.
* Allow referees to log events, sync schedules, receive notifications, and communicate.
* Allow fans to register, follow teams/players, view schedules and live updates, receive notifications, and use the system via mobile in local language and currency.
* Provide a scalable, concurrent communication channel for all stakeholders to receive immediate updates about cancellations and rescheduling.

## Preconditions
None.

## Postconditions
Fan has received the latest live updates and can view the current match state, schedule adjustments, and budget status in their preferred language and currency.

## Main Flow
1. **Fan initiates action** – The fan opens the mobile app or visits the web portal and navigates to the “Live Updates” section.  
2. **System authenticates user** – The system verifies the fan’s credentials and retrieves user preferences (language, currency, followed teams).  
3. **System subscribes fan to relevant streams** – The system registers the fan’s device for push notifications and websockets for the teams and matches they follow.  
4. **Fan receives push notification** – When a match event (e.g., goal, card, substitution) occurs, the IFA’s event feed triggers a push notification.  
5. **System formats payload** – The system translates the event description into the fan’s language, converts any monetary values to the fan’s currency, and attaches a short image or icon.  
6. **Fan opens notification** – The fan taps the notification or views the live feed within the app.  
7. **System displays detailed update** – The app shows the event details, live score, time stamp, and any related context (e.g., player stats, budget impact).  
8. **Fan interacts with update** – The fan can swipe for more events, share the update, or bookmark it for later.  
9. **System logs activity** – The system records that the fan viewed the update for analytics and future personalization.

## Alternative Flows
1. **Fan has no active subscription** –  
   1.1. The fan is prompted to select teams or matches to follow.  
   1.2. Once selected, the system subscribes the fan (see step 3).  

2. **Device offline** –  
   2.1. The fan receives a queued notification when the device reconnects.  
   2.2. The system delivers the missed updates in order of occurrence.  

3. **Fan changes language/currency** –  
   3.1. The fan updates preferences in account settings.  
   3.2. The system immediately re‑formats subsequent notifications and live feed entries to match the new settings.

## Exceptions
1. **Authentication failure** –  
   1.1. The system displays an error message and directs the fan to reset credentials.  

2. **Push notification service unavailable** –  
   2.1. The system falls back to polling the server every 30 seconds for new events.  

3. **Event feed delay** –  
   3.1. The system timestamps the event and displays “Live (delayed)” until the correct time is confirmed.  

4. **Currency conversion error** –  
   4.1. The system uses the last known exchange rate and marks the value with a warning icon.  

5. **Unsupported language** –  
   5.1. The system defaults to the fan’s primary language and logs the missing translation for future updates.