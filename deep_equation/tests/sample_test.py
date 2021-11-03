import unittest
from PIL import Image
from pathlib import Path
from deep_equation.src.deep_equation import predictor


class TestRandomModel(unittest.TestCase):
    def setUp(self) -> None:
        base_path = Path(__file__)
        resources_dir = base_path / "resources"
        self.digit_a = Image.open(resources_dir / "digit_a.png")
        self.digit_b = Image.open(resources_dir / "digit_b.png")

        self.input_imgs_a.extend([
            self.digit_a, self.digit_a, self.digit_b, self.digit_b, self.digit_a
        ])
        self.input_imgs_b.extend([
            self.digit_b, self.digit_b, self.digit_a, self.digit_b, self.digit_a
        ])

    def tearDown(self) -> None:
        self.digit_a.close()
        self.digit_b.close()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.digit_a = None
        self.digit_b = None
        self.input_imgs_a = []
        self.input_imgs_b = []
        self.operators = ['+', '-', '*', '/', '*']

    def test_random_predictor(self):
        """
        Test random prediction outputs. 
        """
        basenet = predictor.RandomModel()

        output = basenet.predict(
            self.input_imgs_a, 
            self.input_imgs_b, 
            operators=self.operators, 
            device='cpu',
        )

        self.validate_output(output)
    
    def test_student_predictor(self):
        """
        Test student prediction outputs. 
        """

        basenet = predictor.StudentModel()

        output = basenet.predict(
            self.input_imgs_a, 
            self.input_imgs_b, 
            operators=self.operators, 
            device='cpu',
        )

        self.validate_output(output)

    def validate_output(self, output):
        """
        Validate output format.
        """

        # Make sure we got one prediction per input_sample
        self.assertEqual(len(output), len(self.input_imgs_a))
        self.assertEqual(len(self.input_imgs_b), len(self.input_imgs_a))
        self.assertEqual(type(output), list)

        # Make sure that that predictions are floats and not other things
        self.assertEqual(type(float(output[0])), float)
        
        # Ensure that the output range is approximately correct
        for out in output:
            self.assertGreaterEqual(out, -10)
            self.assertLessEqual(out, 100)

        # Comparing expected result classes
        self.assertEqual([67.00, 51.00, 82.00, 37.00, 88.00], output)
