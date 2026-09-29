# One robot or the ball, in the field frame: origin at the field centre, x towards the yellow goal,
# y to the left of it, z up (the sim's "field" TF frame).
int32 id          # robot: pattern id, 1-10 blue team, 11-20 yellow (vision/patterns.yaml); ball: 0
bool detected     # false: not seen recently; the other fields hold the last estimate
float64 x         # [m]
float64 y         # [m]
float64 theta     # [rad] heading of the robot's front (the ID-colour half), counter-clockwise from +x
float64 vx        # [m/s] field frame
float64 vy        # [m/s]
float64 w         # [rad/s]
