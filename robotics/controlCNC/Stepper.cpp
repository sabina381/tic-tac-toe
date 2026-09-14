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

// 4095 steps for 360 degree, 8 steps for 5.625 degree

#include "Arduino.h"
#include "StepMotor.h"

Stepper::Stepper(int motor_pin_1, int motor_pin_2, int motor_pin_3, int motor_pin_4) {
    this->step_signal = 0; // signal range (0-7)
    this->direction = 1; // init direction : clock wise
    this->step_delay = 800; // init 
    
    // define pin attributes
    this->motor_pin_1 = motor_pin_1;
    this->motor_pin_2 = motor_pin_2;
    this->motor_pin_3 = motor_pin_3;
    this->motor_pin_4 = motor_pin_4;
    
    // setup pins 
    pinMode(this->motor_pin_1, OUTPUT);
    pinMode(this->motor_pin_2, OUTPUT);
    pinMode(this->motor_pin_3, OUTPUT);
    pinMode(this->motor_pin_4, OUTPUT);
}
void Stepper::setSpeed(long speed)
{
//   this->step_delay = 60L * 1000L * 1000L / this->number_of_steps / speed;
    this->step_delay = speed;
}

void Stepper::step(int n_steps, int direction) { 
    this->direction = direction; // direction 수정 
    // this->step_signal = 0; // 초기화, 듀얼로 움직이는 것 때문에 초기화가 필요하다. 
    n_steps = abs(n_steps); // 혹시 음수 값 넣어도 양수로 전환, 이후에 에러나게 수정하자. 

    unsigned long previous_time; //  = micros();
    unsigned long current_millis;
    long time;

    while (n_steps > 0) {
        current_millis = micros();
        if (current_millis - previous_time >= this->step_delay) {

            this->stepper(); // move single step 
            time += micros() - previous_time;
            previous_time = micros();
            n_steps--;
            delay(1);
        }
    }
}

void Stepper::stepper() {
    switch (this->step_signal) {
        case 0: // 0001
            this->pinSignal(LOW, LOW, LOW, HIGH);
            break;

        case 1: // 0011
            this->pinSignal(LOW, LOW, HIGH, HIGH);
            break;

        case 2: // 0010
            this->pinSignal(LOW, LOW, HIGH, LOW);
            break;

        case 3: // 0110
            this->pinSignal(LOW, HIGH, HIGH, LOW);
            break;

        case 4: // 0100
            this->pinSignal(LOW, HIGH, LOW, LOW);
            break;

        case 5: // 1100
            this->pinSignal(HIGH, HIGH, LOW, LOW);
            break;

        case 6: // 1000
            this->pinSignal(HIGH, LOW, LOW, LOW);
            break;

        case 7: // 1001
            this->pinSignal(HIGH, LOW, LOW, HIGH);
            break;

        default:
            digitalWrite(this->motor_pin_1, LOW);
    }
    this->controlSignal(); // set 
}

void Stepper::pinSignal(int a, int b, int c, int d) {
    digitalWrite(this->motor_pin_1, a);
    digitalWrite(this->motor_pin_2, b);
    digitalWrite(this->motor_pin_3, c);
    digitalWrite(this->motor_pin_4, d);
}

void Stepper::controlSignal() {

    // by using direction & step_signal attribute, 
    // control step_signal sequence.
    // control step_signal by direction. 
    
    if (this->direction == 1) {
        this->step_signal--;
    } else {
        this->step_signal++;
    }
    // keep range 0-7
    if (this->step_signal > 7) {
        this->step_signal = 0;
    }
    if (this->step_signal < 0) {
        this->step_signal = 7;
    }
}

int Stepper::version(void)
{
  return 1;
}

