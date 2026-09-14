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

#include "Arduino.h"
#include "StepMotor.h"

MultiStepper::MultiStepper(Stepper* stepper_x, Stepper* stepper_y)
    :stepper_x(stepper_x), stepper_y(stepper_y) { }

void MultiStepper::setSpeed(long speed) {
    stepper_x->setSpeed(speed);
    stepper_y->setSpeed(speed);
}

void MultiStepper::step(int n_steps, int x_direction, int y_direction) {
    // x, y direction
    stepper_x->direction = x_direction;
    stepper_y->direction = y_direction;

    // total steps 
    n_steps = abs(n_steps);

    unsigned long previous_time; //  = micros()
    unsigned long current_millis;
    long time;

    if (stepper_x->direction == stepper_y->direction) { 

        // Left Up, Right Down 

      while (n_steps > 0) {
        
        current_millis = micros();
        if (current_millis - previous_time >= stepper_x->step_delay) {
            this->moveDiagonalLURD(); // move single step 
            time += micros() - previous_time;
            previous_time = micros();
            n_steps--;
            delay(1);
        }
      }
    } else { 

        // Right Up, Left Down 

      while (n_steps > 0) {
        current_millis = micros();
        if (current_millis - previous_time >= stepper_y->step_delay) {
            this->moveDiagonalRULD(); // move single step 
            time += micros() - previous_time;
            previous_time = micros();
            n_steps--;
            delay(1);
        }
      }
    }
}

void MultiStepper::moveDiagonalLURD() {
    switch (stepper_x->step_signal) {
        case 0: // 0001 0001
            this->pinSignal(LOW, LOW, LOW, HIGH, LOW, LOW, LOW, HIGH);
            stepper_x->step_signal = 0;
            stepper_y->step_signal = 0;
            break;

        case 1: // 0011 0011
            this->pinSignal(LOW, LOW, HIGH, HIGH, LOW, LOW, HIGH, HIGH);
            stepper_x->step_signal = 1;
            stepper_y->step_signal = 1;
            break;

        case 2: // 0010 0010
            this->pinSignal(LOW, LOW, HIGH, LOW, LOW, LOW, HIGH, LOW);
            stepper_x->step_signal = 2;
            stepper_y->step_signal = 2;
            break;

        case 3: // 0110 0110
            this->pinSignal(LOW, HIGH, HIGH, LOW, LOW, HIGH, HIGH, LOW);
            stepper_x->step_signal = 3;
            stepper_y->step_signal = 3;
            break;

        case 4: // 0100 0100
            this->pinSignal(LOW, HIGH, LOW, LOW, LOW, HIGH, LOW, LOW);
            stepper_x->step_signal = 4;
            stepper_y->step_signal = 4;
            break;

        case 5: // 1100 1100
            this->pinSignal(HIGH, HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW);
            stepper_x->step_signal = 5;
            stepper_y->step_signal = 5;
            break;

        case 6: // 1000 1000
            this->pinSignal(HIGH, LOW, LOW, LOW, HIGH, LOW, LOW, LOW);
            stepper_x->step_signal = 6;
            stepper_y->step_signal = 6;
            break;

        case 7: // 1001 1001
            this->pinSignal(HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW, HIGH);
            stepper_x->step_signal = 7;
            stepper_y->step_signal = 7;
            break;

        default:
            digitalWrite(stepper_x->motor_pin_1, LOW);
            digitalWrite(stepper_y->motor_pin_1, LOW);
    }
    this->controlSignal(); // set 
}



void MultiStepper::moveDiagonalRULD() {
    switch (stepper_x->step_signal) { // x signal 기준으로
        case 0: // 0001 1001
            this->pinSignal(LOW, LOW, LOW, HIGH, HIGH, LOW, LOW, HIGH);
            stepper_x->step_signal = 0;
            stepper_y->step_signal = 7;
            break;

        case 1: // 0011 1000
            this->pinSignal(LOW, LOW, HIGH, HIGH, HIGH, LOW, LOW, LOW);
            stepper_x->step_signal = 1;
            stepper_y->step_signal = 6;
            break;

        case 2: // 0010 1100
            this->pinSignal(LOW, LOW, HIGH, LOW, HIGH, HIGH, LOW, LOW);
            stepper_x->step_signal = 2;
            stepper_y->step_signal = 5;
            break;

        case 3: // 0110 0100
            this->pinSignal(LOW, HIGH, HIGH, LOW, LOW, HIGH, LOW, LOW);
            stepper_x->step_signal = 3;
            stepper_y->step_signal = 4;
            break;

        case 4: // 0100 0110
            this->pinSignal(LOW, HIGH, LOW, LOW, LOW, HIGH, HIGH, LOW);
            stepper_x->step_signal = 4;
            stepper_y->step_signal = 3;
            break;

        case 5: // 1100 0010
            this->pinSignal(HIGH, HIGH, LOW, LOW, LOW, LOW, HIGH, LOW);
            stepper_x->step_signal = 5;
            stepper_y->step_signal = 2;
            break;

        case 6: // 1000 0011
            this->pinSignal(HIGH, LOW, LOW, LOW, LOW, LOW, HIGH, HIGH);
            stepper_x->step_signal = 6;
            stepper_y->step_signal = 1;
            break;

        case 7: // 1001 0001
            this->pinSignal(HIGH, LOW, LOW, HIGH, LOW, LOW, LOW, HIGH);
            stepper_x->step_signal = 7;
            stepper_y->step_signal = 0;
            break;

        default:
            digitalWrite(stepper_x->motor_pin_1, LOW);
            digitalWrite(stepper_y->motor_pin_1, LOW);
    }
    this->controlSignal(); // set 
}


void MultiStepper::pinSignal(int a1, int b1, int c1, int d1, int a2, int b2, int c2, int d2){
    // pin signal on stepper_x
    digitalWrite(stepper_x->motor_pin_1, a1);
    digitalWrite(stepper_x->motor_pin_2, b1);
    digitalWrite(stepper_x->motor_pin_3, c1);
    digitalWrite(stepper_x->motor_pin_4, d1);
    // pin signal on stepper_y
    digitalWrite(stepper_y->motor_pin_1, a2);
    digitalWrite(stepper_y->motor_pin_2, b2);
    digitalWrite(stepper_y->motor_pin_3, c2);
    digitalWrite(stepper_y->motor_pin_4, d2);
}

void MultiStepper::controlSignal() {

    // by using direction & step_signal attribute, 
    // control step_signal sequence.
    // control step_signal by direction. 
    
    // x
    if (stepper_x->direction == 1) {
        stepper_x->step_signal--;
    } else {
        stepper_x->step_signal++;
    }
    // keep range 0-7
    if (stepper_x->step_signal > 7) {
        stepper_x->step_signal = 0;
    }
    if (stepper_x->step_signal < 0) {
        stepper_x->step_signal = 7;
    }

    // y
    if (stepper_y->direction == 1) {
        stepper_y->step_signal--;
    } else {
        stepper_y->step_signal++;
    }
    // keep range 0-7
    if (stepper_y->step_signal > 7) {
        stepper_y->step_signal = 0;
    }
    if (stepper_y->step_signal < 0) {
        stepper_y->step_signal = 7;
    }
}

int MultiStepper::version(void)
{
  return 1;
}