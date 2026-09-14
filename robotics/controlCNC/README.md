## Structure
```
StepMotor 
 - StepMotor.h
 - Stepper.cpp
 - MultiStepper.cpp
 - TicTacToeArtist.cpp
 - controller
   - controller
```

## Scripts with Key Methods 
```cpp
#include "StepMotor.h"

Stepper stepper_x(8, 9, 10, 11);
Stepper stepper_y(3, 4, 5, 6);
Stepper stepper_z(A5, A4, A3, A2);

TicTacToeArtist artist(21, 21, &stepper_x, &stepper_y, &stepper_z);

void setup(){
    Serial.begin(115200);
    Serial.println("Setup complete");
};

void loop() {
    Serial.println("delay");
    delay(3000);
    artist.resetPosition();
    artist.drawGameBoard();   // 게임 보드 그리기
    delay(2000);              // 잠시 대기

    Serial.println("Drawing X...");
    artist.drawX(0);           // 'X' 그리기
    delay(2000);              // 잠시 대기
    
    Serial.println("Drawing Circle...");
    artist.drawCircle(0);      // 'O' 그리기
    delay(2000);              // 잠시 대기


    Serial.println("end");
    delay(1000000);
};
```

- `drawGameBoard`  
- `drawX(position)`  
- `drawCircle(position)`


## Target Step motor 
name : 28BYJ-48(5V) motor  
link : [AliExpress](https://ko.aliexpress.com/item/1005006141719157.html?spm=a2g0o.productlist.main.1.64ee439aCk2cQE&algo_pvid=0028234f-f163-40ff-af87-7c5225081d42&aem_p4p_detail=202409110014239502571167064400008634737&algo_exp_id=0028234f-f163-40ff-af87-7c5225081d42-0&pdp_npi=4%40dis%21KRW%212680%212680%21%21%2113.85%2113.85%21%402101584517260388637604119ea2ac%2112000035947534362%21sea%21KR%210%21ABX&curPageLogUid=SUfcCKzo8Wsh&utparam-url=scene%3Asearch%7Cquery_from%3A&search_p4p_id=202409110014239502571167064400008634737_1)

## Reference
1.  [아두이노 중급_29. 스텝모터,스테핑모터](https://m.blog.naver.com/PostView.naver?blogId=darknisia&logNo=221652111026&proxyReferer=https:%2F%2Fwww.google.com%2F&trackingCode=external)
2.  [http://www.arduino.cc/en/Reference/Stepper](http://www.arduino.cc/en/Reference/Stepper)
