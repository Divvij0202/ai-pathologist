# Failure gallery (standin, best_first, budget 40)

Green outline: reference tumour. Tinted tiles: classifier probability >= 0.5 (exhaustive scan).

## 1. test_040: missed lesion

2083 um lesion; classifier does fire on it when viewed. Likely cause: search failure: the classifier sees it, the search never looked there.

![](01_test_040_missed_lesion.jpg)

## 2. test_082: missed lesion

1762 um lesion; classifier does not fire on it when viewed. Likely cause: model failure: the classifier does not fire on it.

![](02_test_082_missed_lesion.jpg)

## 3. test_075: missed lesion

1542 um lesion; classifier does fire on it when viewed. Likely cause: search failure: the classifier sees it, the search never looked there.

![](03_test_075_missed_lesion.jpg)

## 4. test_001: missed lesion

1401 um lesion; classifier does fire on it when viewed. Likely cause: search failure: the classifier sees it, the search never looked there.

![](04_test_001_missed_lesion.jpg)

## 5. test_075: missed lesion

916 um lesion; classifier does fire on it when viewed. Likely cause: search failure: the classifier sees it, the search never looked there.

![](05_test_075_missed_lesion.jpg)

## 6. test_044: false alarm

909 tiles >= 0.5 on a normal slide. Likely cause: model failure: tissue mistaken for tumour.

![](06_test_044_false_alarm.jpg)

## 7. test_098: false alarm

426 tiles >= 0.5 on a normal slide. Likely cause: model failure: tissue mistaken for tumour.

![](07_test_098_false_alarm.jpg)

## 8. test_055: false alarm

209 tiles >= 0.5 on a normal slide. Likely cause: model failure: tissue mistaken for tumour.

![](08_test_055_false_alarm.jpg)

## 9. test_034: false alarm

208 tiles >= 0.5 on a normal slide. Likely cause: model failure: tissue mistaken for tumour.

![](09_test_034_false_alarm.jpg)

## 10. test_037: false alarm

133 tiles >= 0.5 on a normal slide. Likely cause: model failure: tissue mistaken for tumour.

![](10_test_037_false_alarm.jpg)
