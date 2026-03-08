# BUT_Parking_Detection_CTM-and-LB
Parking Project Cyriaque T. M. and Louis B.

# New fonctionnality - [license_plate_anonymization](https://github.com/LouisB4455/BUT_Parking_Detection_CTM-and-LB/blob/main/license_plate_anonymization)
Added on March 8, 2026: a function capable of automatically blurring vehicle license plates.
This function can then be combined with a webcam for image capture every minutes (input of the function) and automatic upload of blurred photos to Git Hub (output of the function).

**Scheme :**
Material: Raspberry Pi + Webcam
➜ Every minute, the Raspberry Pi takes a photo using the webcam.
➜ License plates are automatically detected and blurred.
➜ A user validation step ensures data confidentiality.
➜ After validation, the images are automatically uploaded to GitHub.
➜ The blurred images are then processed to determine the number of available parking spaces or the number of occupied spaces in the parking lot. This process can be performed on the Raspberry Pi, on a personal computer, or on a server.
