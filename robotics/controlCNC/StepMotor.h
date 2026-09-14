// This StepMotor Library is for 28BYJ-48(5V) Step Motor.
// UniPolar 5 wire, half-step sequence

// Switching Seq 
// Step  | IN1 IN2 IN3 IN4
// ----- | ----------------
//   0   |  0   0   0   1
//   1   |  0   0   1   1
//   2   |  0   0   1   0
//   3   |  0   1   1   0
//   4   |  0   1   0   0
//   5   |  1   1   0   0
//   6   |  1   0   0   0
//   7   |  1   0   0   1

// [ Coding Rule ]
// methods naming : Camel Casing
// attribute naming : Snake Casing

#ifndef StepMotor_h
#define StepMotor_h

class Stepper {
public:
    Stepper(int motor_pin_1, int motor_pin_2, int motor_pin_3, int motor_pin_4);

    void setSpeed(long speed);
    void step(int n_steps, int direction); 
    int version(void);

    void stepper();               // return signal per position
    void controlSignal();          // control steps by direction and defined range of step_signal
    void pinSignal(int a, int b, int c, int d); // pin step signal 
    void reset();

    // motor pin numbers
    int motor_pin_1;
    int motor_pin_2;
    int motor_pin_3;
    int motor_pin_4;

    // movement information
    int step_signal;    // current step signal; range 0-7
    int direction;      // 0: clockwise, 1: counterclockwise

    int number_of_steps = 4096; // 한 바퀴를 돌 때 
    long step_delay = 800L; // 속도 조절을 위한 값 

};

class MultiStepper {
private:             
    Stepper* stepper_x;
    Stepper* stepper_y;

public:
    MultiStepper(Stepper* stepper_x, Stepper* stepper_y);

    void setSpeed(long speed);
    void step(int n_steps, int x_direction, int y_direction);
    int version(void);

    void moveDiagonalLURD(); // Left UP, Right Down 
    void moveDiagonalRULD(); // Right Up, Left Down
    void pinSignal(int a1, int b1, int c1, int d1, int a2, int b2, int c2, int d2);
    void controlSignal();
    
};

// drawing class  
class TicTacToeArtist {
private:
    // define stepper motors
    Stepper* stepper_x;
    Stepper* stepper_y; 
    Stepper* stepper_z;
    MultiStepper multi_stepper;

    // define board size 
    int board_width_grid;
    int board_height_grid;

    // cal cell size
    int cell_width_grid;
    int cell_height_grid;
    
    // X 변의 길이, 팔각형 변의 길이 
    int x_grid_line;
    int circle_grid_line;

    // stepper settings
    int steps_per_grid = 423; // 1모눈에 423 steps
    

public:

    TicTacToeArtist(int board_width_grid, int board_height_grid, Stepper* stepper_x, Stepper* stepper_y, Stepper* stepper_z);

    int cumulated_step_x;
    int cumulated_step_y;
    
    // drawing methods
    void drawCircle(int position);
    void drawX(int position);
    void drawGameBoard();

    // control pen
    void penUp();                    // lift pen to stop drawing
    void penDown();                  // put down pen to start drawing

    // method for positions
    void resetPosition();            // return to position (0,0)
    void setPosition(int position);
    void move(int x_grids, int y_grids, int x_direction, int y_direction); // move to (x, y)
    void moveDiagonally(int n_grids, int x_direction, int y_direction);
    void moveX(int n_grids, int direction);
    void moveY(int n_grids, int direction);
    
};

#endif