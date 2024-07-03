| name                         | type      | description                                         | comment                            |
|:-----------------------------|:----------|:----------------------------------------------------|:-----------------------------------|
| trip_id                      | Integer   | מפתח נסיעה                                          |                                    |
| part                         | text      | מחובר לtrip id ומרכיב את trip_id_unique             |                                    |
| trip_id_unique_station       | Text(key) | trip_id_unique+station_index                        | מקושר לשאלונים, ספירות בתחנה/נסיעה |
| trip_id_unique               | Text(key) | קוד נסיעה                                           | מקושר לשאלונים, ספירות בתחנה/נסיעה |
| line_id                      | Integer   | קוד קו                                              |                                    |
| direction                    | Integer   | כיוון                                               |                                    |
| alternative                  | Text      | חלופה                                               |                                    |
| cluster                      | Text      | שם אשכול                                            |                                    |
| station_index                | Integer   | מספר סידורי של תחנה                                 | בסדר עולה - ניתן לשרטט מסלול קו    |
| station_id                   | Integer   | קוד תחנה תואם משרד התחבורה                          |                                    |
| station_name                 | Text      | שם התחנה                                            |                                    |
| arrival_time                 | Time      | שעת הגעת האוטובוס לתחנה                             | hh:mm                              |
| door_closing_time            | Time      | שעת סגירת הדלת                                      | hh:mm                              |
| arrival_is_estimated         | Bool      | האם שעת הגעת האוטובוס לתחנה משוערכת, או נמדדה בדיוק |                                    |
| latitude                     | Real      | קו רוחב -תחנה                                       |                                    |
| longitude                    | Real      | קו אורך - תחנה                                      |                                    |
| passengers_up                | Integer   | מספר נוסעים שעלו                                    |                                    |
| passengers_continue          | Integer   | מספר נוסעים ממשיכים                                 |                                    |
| mekadem_nipuach_luz          | Real      | מקדם ניפוח לוז                                      |                                    |
| passengers_continue_menupach | Real      | ממשיכים מנופח                                       |                                    |