use utils::stat;

pub struct LinearRegression {
    pub coefficient: Option<f32>,
    pub intercept: Option<f32>
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            coefficient: None, intercept: None
        }
    }

    pub fn fit(&mut self, x_vals: &Vec<f32>, y_vals: &Vec<f32>) {
        let b1 = stat::covariance(x_vals, y_vals) / stat::variance(x_vals);
        let b0 = stat::mean(y_vals) - b1 * stat::mean(x_vals);

        self.intercept = Some(b0);
        self.coefficient = Some(b1);
    }

    pub fn predict(&self, x: f32) -> f32 {
        if self.coefficient.is_none() || self.intercept.is_none() {
            panic!("fit(..) must be called first");
        }

        let b0 = self.intercept.unwrap();
        let b1 = self.coefficient.unwrap();

        return b0 + b1 * x;
    }

    pub fn predict_list(&self, x_vals: &Vec<f32>) -> {
        let mut predictions = Vec::new();

        for i in 0..x_vals.len() {
            predictions.push(self.predict(x_vals[i]));
        }

        return predictions;
    }

    fn root_mean_squared_error(&self, actual: &Vec<f32>, predicted: &Vec<f32>) -> {
        let mut sum_error = 0f32;
        let length = actual.len();

        for i in 0..length {
            sum_error += f32::powf(predicted[i] - actual[i], 2f32);
        }

        let mean_error = sum_error / length as f32;
        return mean_error.sqrt();
    }

    pub fn evaluate(&self, x_test: &Vec<f32>, y_test: &Vec<f32>) -> f32 {
        if self.coefficient.is_none() || self.intercept.is_none() {
            panic!("fit(..) must be called first");
        }

        let y_pred = self.predict_list(x_test);
        return self.root_mean_squared_error(y_test, &y_pred);
    }

}
