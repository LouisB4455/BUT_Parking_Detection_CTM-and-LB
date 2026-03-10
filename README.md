# BUT_Parking_Detection_CTM-and-LB
Parking Project Cyriaque T. M. and Louis B.

# New fonctionnality - [license_plate_anonymization](https://github.com/LouisB4455/BUT_Parking_Detection_CTM-and-LB/blob/main/license_plate_anonymization)
Added on March 8, 2026: a function capable of automatically blurring vehicle license plates.
This function can then be combined with a webcam for image capture every minutes (input of the function) and automatic upload of blurred photos to Git Hub (output of the function).

**Scheme :**
Proposed Technical Architecture : 
Material: Raspberry Pi + Webcam
➜ Every minute, the Raspberry Pi takes a photo using the webcam.
➜ License plates are automatically detected and blurred.
➜ A user validation step ensures data confidentiality.
➜ After validation, the images are automatically uploaded to GitHub.
➜ The blurred images are then processed to determine the number of available parking spaces or the number of occupied spaces in the parking lot. This process can be performed on the Raspberry Pi, on a personal computer, or on a server.

# New fonctionnality - [manual_configuration_of_the_parking](https://github.com/LouisB4455/BUT_Parking_Detection_CTM-and-LB/blob/main/config_parking_via_creation_manuelle_polygone.py)
Added on March 9, 2026: a function where the user selects the parking spots they want to detect using a photo of the parking lot. They choose 4 points, and the points are automatically connected. They can right-click to cancel a created purple parking spot. To save, they press the 'S' key on the keyboard. The output of this function is a **parking_slots.pkl** who store the coordinate of all the parking slot. 

# New fonctionnality - [detect_free_parking_slot_based_on_the_config_parking_file](https://github.com/LouisB4455/BUT_Parking_Detection_CTM-and-LB/blob/main/detection_de_place_de_parking_libre_via_ML_YOLO_with_config_parking_file.py)
Added on March 9, 2026: a function that uses YOLO to detect the presence of a vehicle and checks if the detected vehicle occupies at least 30% of the selected parking spot. If so, it counts the spot as occupied. A reformuler car bizarrement formulé.
