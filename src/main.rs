extern crate plotlib;

use mlm::reg::linear_reg;

use plotlib::scatter::Scatter;
use plotlib::scatter;
use plotlib::style::{Marker, Point};
use plotlib::view::View;
use plotlib::page::Page;

fn main() {
    let mut model = linear_reg::LinearRegression::new();
    let x_vals = vec![1f32, 2f32, 3f32, 4f32, 5f32];
    let y_vals = vec![1f32, 1f32, 2f32, 3f32, 5f32];

    model.fit(&x_vals, &y_vals);

    let y_preds: Vec<f32> = model.predict_list(&x_vals);

    let mut actual: Vec<(f64, f64)> = Vec::new();
    let mut prediction: Vec<(f64, f64)> = Vec::new();

    for i in 0..x_vals.len() {
        actual.push((x_vals[i] as f64, y_vals[i] as f64));
        prediction.push((x_vals[i] as f64, y_preds[i] as f64))
    }

    let plot_actual = Scatter::from_vec(&actual)
        .style(scatter::Style::new()
        .colour("#e41a1c"));

    let plot_pred = Scatter::from_vec(&prediction)
        .style(scatter::Style::new()
        .marker(Marker::Square)
        .colour("#377eb8"));

    let view = View::new()
        .add(&plot_actual)
        .add(&plot_pred)
        .x_range(-0., 6.)
        .y_range(0., 6.)
        .x_label("x")
        .y_label("y");

    Page::single(&view).save("scatter.svg");

    println!("Coefficient: {0}", model.coefficient.unwrap());
    println!("Intercept: {0}", model.intercept.unwrap());
    println!("Accuracy: {0}", model.evaluate(&x_vals, &y_vals));
}
