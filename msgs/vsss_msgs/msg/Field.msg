# The global game state: everything the strategy needs, from the vision (/field) or the sim's ground
# truth (/field_truth). header.stamp is the time of the camera frame (or sim state) it describes, so
# consumers can tell how old it is; header.frame_id is "field".
std_msgs/Header header
Object ball
Object[] blue     # robots of the blue team, by id
Object[] yellow   # robots of the yellow team, by id
