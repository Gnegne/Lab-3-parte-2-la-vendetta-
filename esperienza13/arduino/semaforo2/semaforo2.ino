// prova n.1 di macchina a stati finiti semaforo con buzzer che sona su LED giallo e LED verde accesi
//non ancora verificato
//avtore: Thomas Giannoni
//vltima modifica: 26/04/2017

#define INP 8   //il nostro abilitatore
#define LED1 9  //LED verde  
#define LED2 10  //LED giallo
#define LED3 11  //LED rosso
#define buzzer 12 // per farlo sonare collegare altra estremità a GND

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(INP, INPUT_PULLUP);
  pinMode(buzzer, OUTPUT);
}

void loop() {
  // le variabili che ho usato
  int val = 0;
  int t = 0;
  val = digitalRead(INP);
  if(val == HIGH) {
    while(t<1000){
      digitalWrite(LED2, HIGH);
      delay(100);
      digitalWrite(LED2, LOW);
      delay(100);
      t = t + 200;
    }
    t = 0;
   }
   else{
    digitalWrite(LED1, HIGH);
    delay(1000);
    val = digitalRead(INP);
    if(val == HIGH) {
        digitalWrite(LED1, LOW);
        return;
    }
    digitalWrite(LED2, HIGH);
    while(t<1000){
      tone(buzzer,3000,200);
      delay(100);
      noTone(buzzer);
      delay(100);
      t = t + 200;
      }
      t = 0;
    }
    val = digitalRead(INP);
    digitalWrite(LED1, LOW);
    digitalWrite(LED1, LOW);
     if(val == HIGH) {
      return;
    }
    digitalWrite(LED3, HIGH);
    delay(1000);
    digitalWrite(LED3, LOW);
   }


