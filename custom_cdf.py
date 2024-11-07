"""
Custom CDF class

Holds distribution information

"""

import numpy as np

class customCDF:
    def __init__(self, num_inputs=5, min=0, max=1):
        assert num_inputs > 2, "Number of inputs must be greater than 2."
        self.num_inputs = num_inputs
        self.min = float(min)
        self.max = float(max)
        self.x_points = np.linspace(min, max, self.num_inputs, endpoint=True)
        self.y_points = np.linspace(0, 1, self.num_inputs, endpoint=True)
    
    def getCDF(self):
        return list(zip(self.x_points, self.y_points))
    
    def xvalues(self):
        return self.x_points

    def yvalues(self):
        return self.y_points
    
    def _pdf_slopes(self):
        return (self.y_points[1:] - self.y_points[:-1]) / (self.x_points[1:] - self.x_points[:-1])
    
    def mean(self):
        pdf_slopes = self._pdf_slopes()
        mean = sum(pdf_slopes[i] * (self.x_points[i+1]**2 - self.x_points[i]**2) / 2 for i in range(len(pdf_slopes)))
        return mean

    def mean_squre(self):
        pdf_slopes = self._pdf_slopes()
        m2 = sum(pdf_slopes[i] * (self.x_points[i+1]**3 - self.x_points[i]**3) / 3 for i in range(len(pdf_slopes)))
        return m2
    
    def stdev(self):
        m = self.mean()
        m2 = self.mean_squre()
        variance = m2 - m**2
        return variance**0.5

    def inv_sample(self, y: float) -> float:
        """Inverse sample the distribution, returning an x-value"""
        return np.interp(y, self.y_points, self.x_points)

    def sample(self, x: float) -> float:
        """Sample the distribution, returning a y-value"""
        print(f"SAMPLE: {str(x)}")
        res = np.interp(x, self.x_points, self.y_points)
        print(f"xpts: {str(self.x_points)}")
        print(f"ypts: {str(self.y_points)}")
        print(f"RES: {str(res)}")
        return np.interp(x, self.x_points, self.y_points)

    def set_y_by_value(self, x_value, y_value):
        """Update the y_value and ensure the CDF remains valid."""
        try:
            idx = np.where(np.isclose(self.x_points, x_value))
            idx = idx[0].item()
        except:
            raise ValueError(f"Invalid x_value provided. X Points are {str(self.x_points)}")
        
        self.set_y_by_index(idx, y_value)
        
    def set_y_by_index(self, idx, y_value):
        """Update the y value based on the index."""
        
        if idx > 0 and self.y_points[idx-1] < y_value:
            self.y_points[idx] = y_value
        elif idx == 0:
            self.y_points[idx] = y_value
        
        # Ensure distribution never decreases, since CDF
        self.y_points[idx:] = np.maximum(self.y_points[idx:], y_value)

        # always end at 1 for CDF
        self.y_points[-1] = 1
        
    def __len__(self):
        return self.num_inputs