# Smart Coin Classifier

The Smart Coin Classifier is an automated system designed to separate Indian currency coins using image processing and a mechanical arrangement driven by a servo motor. It leverages the SIFT (Scale-Invariant Feature Transform) algorithm to detect and describe local features in images, ensuring accurate coin classification by comparing test and training data.

## Output(working_model_video)
https://drive.google.com/file/d/1h8Hmvu1AXPrm8oryWmI_4geReZQwES3H/view?usp=sharing

## Features

- **Image Processing with SIFT**: Uses the SIFT algorithm for precise coin detection and classification.
- **Mechanical Sorting**: Employs a servo motor to sort coins into appropriate bins based on classification.
- **Automation**: Fully automated system from detection to sorting.
- **High Accuracy**: Reliable and consistent coin classification and sorting.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- RasberryPi

### Usage

1. **Run the main script**:

   ```bash
   python coin_match_final.py
   ```

2. **Place coins**: Position the coins in the designated area for image capture.

3. **Start Sorting**: The system will classify the coins using the SIFT algorithm and sort them using the servo motor mechanism.

## Folder Structure

```
Smart-Coin-Classifier/
│
├── query image/             # Images used for querying
│
├── test/                    # Test images
│
├── coin_match.py            # Initial coin matching script
├── coin_match_final.py      # Final coin matching script
├── coin matching try.py     # Trial script for coin matching
├── servo.py                 # Script for controlling the servo motor
├── structure.jpg            # Image depicting the structure of the setup
├── Result.PNG               # Result image
```

Feel free to contact us for any queries or support.

Happy Coin Collecting!
---

# Contact

- **Name**: Shiraz Anwar
- **Email**: kshirazanwar@gmail.com
- **GitHub**: [shirazanwar97](https://github.com/shirazanwar97)
