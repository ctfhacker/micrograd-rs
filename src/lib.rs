#![feature(maybe_uninit_uninit_array)]

use std::collections::{BTreeSet, VecDeque};
use std::path::Path;

mod rng;
use rng::Rng;
use rand::Rng as RngTrait;

#[derive(Debug)]
pub enum Error {
    /// Operation has invalid children during back propagation
    InvalidChildren,
}

/// Result type for this crate
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Default)]
pub struct Expression {
    /// Random Number Generator for this expression
    rng: Rng,

    /// Current values known by this expression
    values: Vec<f64>,

    /// The indexes of the children
    children: Vec<Option<Vec<ValueId>>>,

    /// The operation performed on the `children` to result in this data
    operation: Vec<Option<Operation>>,

    /// Labels for the values in this expression
    labels: Vec<String>,

    /// Gradients of the values in this expression
    gradients: Vec<f64>,
}

impl Expression {
    pub fn get_value(&self, node: &ValueId) -> f64 {
        self.values[node.0]
    }

    pub fn as_dot(&self, outfile: &str) {
        let mut dot = String::new();
        dot.push_str("digraph {\n");
        dot.push_str("  rankdir=\"TB\"\n");
        for i in 0..self.len() {
            dot.push_str(&format!(
                "  node_{i} [label=\"{}\ndata {:.4?} grad {:.4?}\"]\n",
                self.labels[i], self.values[i], self.gradients[i]
            ));
            if let Some(children) = &self.children[i] {
                let operation = self.operation[i].expect("Children with no operation?");

                dot.push_str(&format!("  op_{i} [label=\"{operation:?}\"]\n"));

                for child in children {
                    dot.push_str(&format!("  node_{} -> op_{i}\n", child.0));
                }

                dot.push_str(&format!("  op_{i} -> node_{i}\n"));
            }
        }
        dot.push_str("}\n");

        let outfile = Path::new(outfile);
        std::fs::write(outfile, dot).unwrap();

        if let Ok(data) = std::process::Command::new("dot")
            .args(["-Tsvg", outfile.as_os_str().to_str().unwrap()])
            .output()
        {
            std::fs::write(outfile.with_extension("svg"), data.stdout).unwrap();
        }
    }

    /// Nudge all values by the given `nudge_by` distance
    /// 
    /// Updates the value of each node to be `value += nudge_by * curr_gradient`
    pub fn nudge_by(&mut self, nudge_by: f64) {
        assert!(nudge_by.is_sign_negative());

        for (value, gradient) in self.values.iter_mut().zip(self.gradients.iter()) {
            *value += nudge_by * gradient;
        }
    }

    /// Zero out the gradients
    pub fn zero_grad(&mut self) {
        self.gradients.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// The operation performed which resulted in a Value
#[derive(Clone, Copy)]
pub enum Operation {
    Add,
    Multiply,
    Tanh,
    Exp,
    Power
}

impl std::fmt::Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        match self {
            Operation::Add => write!(f, "+"),
            Operation::Multiply => write!(f, "*"),
            Operation::Tanh => write!(f, "tanh"),
            Operation::Exp => write!(f, "exp"),
            Operation::Power => write!(f, "^"),
        }
    }
}

#[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
pub struct ValueId(usize);

impl Expression {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A single neuron with `INPUTS` inputs
#[derive(Debug, Clone)]
pub struct Neuron {
    /// Weights for this neuron
    weights: Vec<ValueId>,

    /// The bias of this neuron
    bias: ValueId
}

/// A single layer of `NEURONS` neurons, each with `INPUTS` inputs
#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>
}

/// A collection of layers into an MLP
pub struct MultilayerPerceptron {
    layers: Vec<Layer>
}

impl Expression {
    /// Create a new value in the expression with `data` and named `label`
    pub fn new_value(&mut self, label: &str, data: f64) -> ValueId {
        // Get the id for this value
        let id = self.values.len();

        // Add this value to the current expression
        self.values.push(data);
        self.children.push(None);
        self.operation.push(None);
        self.labels.push(label.to_string());
        self.gradients.push(0.0);

        // Return the id of this value
        ValueId(id)
    }

    /// Create a new value in the expression with `data` and named `label` with the given `children` 
    /// as the given [`Operation`]
    pub fn new_value_with_children(
        &mut self,
        label: &str,
        data: f64,
        children: [Option<ValueId>; 2],
        operation: Operation,
    ) -> ValueId {
        // Create the children for this expression
        let children = children.iter().flatten().copied().collect();

        // Get the id for the new value
        let id = self.values.len();

        // Add the new value
        self.values.push(data);
        self.children.push(Some(children));
        self.operation.push(Some(operation));
        self.labels.push(label.to_string());
        self.gradients.push(0.0);

        // Return the id of the newly created Value
        ValueId(id)
    }

    /// Create a new neuron with `INPUTS` values
    pub fn new_neuron(&mut self, inputs: usize) -> Neuron {
        let mut weights = Vec::new();

        for _ in 0..inputs {
            let rand_value = self.rng.gen_range(-1.0..1.0);
            weights.push(self.new_value("w", rand_value));
        }

        let rand_value = self.rng.gen_range(-1.0..1.0);
        let bias = self.new_value("b", rand_value);
        Neuron {
            weights,
            bias
        }
    }

    /// Activate the given `neuron` with the given `data`
    pub fn activate_neuron(&mut self, neuron: &Neuron, data: &[ValueId]) -> ValueId {
        assert!(neuron.weights.len() == data.len(), "Size mismatch between data and neuron");


        let mut prods = Vec::new();

        // Calculate the weight * datax for each neuron
        for (index, (weight, datax)) in neuron.weights.iter().zip(data.iter()).enumerate() {
            let tmp = self.mul(&format!("w{index}x{index}"), *weight, *datax);
            prods.push(tmp);
        }

        // Sum all of the weights together
        let mut sum = neuron.bias;
        for (index, prod) in prods.iter().enumerate() {
            sum = self.add(&format!("{index}:wx+b"), sum, *prod);
        }

        self.tanh("neuron_tanh", sum)
    }

    /// Create a new layer with `num_inputs` inputs and `num_outputs` outputs
    pub fn new_layer(&mut self, num_inputs: usize, num_outputs: usize) -> Layer {
        let mut neurons = Vec::new();

        // Create a neuron for each output
        for _ in 0..num_outputs {
            neurons.push(self.new_neuron(num_inputs));
        }

        Layer { neurons }
    }

    /// Activate the given layer using the given input `data`
    pub fn activate_layer(&mut self, layer: &Layer, data: &[ValueId]) -> Vec<ValueId> {
        let mut output = Vec::new();

        for neuron in layer.neurons.iter() {
            output.push(self.activate_neuron(neuron, data));
        }

        output
    }

    /// Create a new `MultilayerPerceptron` with the given `inputs` and the given `layer sizes`
    pub fn new_mlp(&mut self, inputs: usize, layer_sizes: &[usize])  -> MultilayerPerceptron {
        // Create a list of all layer sizes
        let mut sizes = vec![inputs];
        sizes.extend_from_slice(layer_sizes);


        let mut layers = Vec::new();

        // For each pair of sizes, create a layer
        // i.e. [3, 4, 1]
        //
        // And they are interconntected
        // * - * - *
        // * x * x
        // * x * x
        //   \ * /
        for i in 0..sizes.len() - 1 {
            layers.push(self.new_layer(sizes[i], sizes[i + 1]));
        }

        // Create the first input layer
        MultilayerPerceptron { layers }
    }

    /// Perform a forward pass of a [`MultilayerPerceptron`] with the given `input_data`
    pub fn activate_mlp(&mut self, mlp: &MultilayerPerceptron, input_data: &[f64]) -> Vec<ValueId> {
        // Initialize the input data
        let mut data = Vec::new();

        for (i, item) in input_data.iter().enumerate() {
            data.push(self.new_value(&format!("input{i}"), *item));
        }

        // Starting with the initial data, hand the data to the first layer for activation
        // Set the result of this layer to data to be used for each subsequent layer
        for layer in mlp.layers.iter() {
            data = self.activate_layer(layer, &data);
        }

        data
    }

    /// Backpropagate from the given value
    pub fn back_prop(&mut self, from: ValueId) -> Result<()> {
        // Clear the gradients
        self.gradients.iter_mut().for_each(|x| *x = 0.0);

        // Initialize the work queue
        let mut data = VecDeque::new();
        data.push_back(from);

        // The derivative of the starting node is always 1.0
        self.gradients[from.0] = 1.0;

        let mut seen = BTreeSet::new();

        while let Some(curr_node) = data.pop_front() {
            // Only visit each node once
            if seen.contains(&curr_node) {
                continue;
            }

            let curr_grad = self.gradients[curr_node.0];

            match self.operation[curr_node.0] {
                Some(Operation::Add) => {
                    let mut iter = self.children[curr_node.0].iter().flatten();

                    let Some(left) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };
                    let Some(right) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };

                    // Add operation always propogates the current gradient unchanged
                    // as per the chain rule
                    self.gradients[left.0] += curr_grad;
                    self.gradients[right.0] += curr_grad;

                    // Add the child nodes to continue back propogating
                    data.push_back(*left);
                    data.push_back(*right);
                }
                Some(Operation::Multiply) => {
                    let mut iter = self.children[curr_node.0].iter().flatten();
                    let Some(left) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };
                    let Some(right) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };

                    // Add operation always propogates the current gradient unchanged
                    self.gradients[left.0] += curr_grad * self.values[right.0];
                    self.gradients[right.0] += curr_grad * self.values[left.0];

                    // Add the child nodes to continue back propogating
                    data.push_back(*left);
                    data.push_back(*right);
                }
                Some(Operation::Tanh) => {
                    let mut iter = self.children[curr_node.0].iter().flatten();
                    let Some(node) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };

                    // Apply the derivative of tanh for the current gradient
                    self.gradients[node.0] += 1.0 - self.values[curr_node.0].powi(2) * curr_grad;

                    // Add the child nodes to continue back propogating
                    data.push_back(*node);
                }
                Some(Operation::Exp) => {
                    let mut iter = self.children[curr_node.0].iter().flatten();
                    let Some(node) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };

                    // Derivative of e^x is e^x 
                    // Continue chain rule from the current gradient
                    self.gradients[node.0] += self.values[node.0] * curr_grad;

                    // Add the child nodes to continue back propogating
                    data.push_back(*node);
                }
                Some(Operation::Power) => {
                    let mut iter = self.children[curr_node.0].iter().flatten();
                    let Some(node) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };
                    let Some(exponent) = iter.next() else { 
                        return Err(Error::InvalidChildren); 
                    };

                    // Calculate the local derivative
                    let local_der = self.values[exponent.0] * self.values[node.0].powf(self.values[exponent.0] - 1.0);

                    // Continue chain rule from the current gradient
                    self.gradients[node.0] += local_der * curr_grad;

                    // Add the child nodes to continue back propogating
                    data.push_back(*node);
                    data.push_back(*exponent);
                }
                None => {
                    // Nothing to back propagate
                }
            }

            // Mark this node as seen
            seen.insert(curr_node);
        }

        Ok(())
    }

    /// Add values `left` and `right` and return the resulting `ValueId`
    pub fn add(&mut self, label: &str, left: ValueId, right: ValueId) -> ValueId {
        let data = self.values[left.0] + self.values[right.0];
        self.new_value_with_children(label, data, [Some(left), Some(right)], Operation::Add)
    }

    /// Negate the `left` value
    pub fn neg(&mut self, label: &str, left: ValueId) -> ValueId {
        let tmp_neg1 = self.new_value(&format!("{label}_neg1"), -1.0);
        self.mul(label, left, tmp_neg1)
    }

    /// Subtract values `left` and `right`
    pub fn sub(&mut self, label: &str, left: ValueId, right: ValueId) -> ValueId {
        let tmp_val = self.neg(&format!("{label}_neg"), right);
        self.add(label, left, tmp_val)
    }

    /// Divide `left` / `right`
    pub fn div(&mut self, label: &str, left: ValueId, right: ValueId) -> ValueId {
        let tmp_neg1 = self.new_value(&format!("{}_neg1", self.labels[right.0]), -1.0);
        let tmp_val = self.pow(&format!("{label}_tmp"), right, tmp_neg1);
        self.mul(label, left, tmp_val)
    }

    /// Multiply values `left` and `right` and return the resulting `ValueId`
    pub fn mul(&mut self, label: &str, left: ValueId, right: ValueId) -> ValueId {
        let data = self.values[left.0] * self.values[right.0];
        self.new_value_with_children(label, data, [Some(left), Some(right)], Operation::Multiply)
    }

    /// Hyperbolic tangent function for `left`
    pub fn tanh(&mut self, label: &str, left: ValueId) -> ValueId {
        let data = self.values[left.0].tanh();
        self.new_value_with_children(label, data, [Some(left), None], Operation::Tanh)
    }

    /// e ^ left
    pub fn exp(&mut self, label: &str, left: ValueId) -> ValueId {
        let data = self.values[left.0].exp();
        self.new_value_with_children(label, data, [Some(left), None], Operation::Exp)
    }

    /// left ^ right
    pub fn pow(&mut self, label: &str, left: ValueId, right: ValueId) -> ValueId {
        let data = self.values[left.0].powf(self.values[right.0]);
        self.new_value_with_children(label, data, [Some(left), Some(right)], Operation::Power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works_easy() {
        let mut expr = Expression::default();

        // Inputs x1 and x2
        let x1 = expr.new_value("x1", 2.0);
        let x2 = expr.new_value("x2", 0.0);

        // Weights w1 and w2
        let w1 = expr.new_value("w1", -3.0);
        let w2 = expr.new_value("w1", 1.0);

        // Bias of a neuron
        let bias = expr.new_value("bias", 6.7);

        // x1*w1 + x2*w2 + bias
        let x1w1 = expr.mul("x1w1", x1, w1);
        let x2w2 = expr.mul("x2w2", x2, w2);
        let x1w1x2w2 = expr.add("x1w1x2w2", x1w1, x2w2);
        let n = expr.add("n", x1w1x2w2, bias);

        // Back prop
        expr.back_prop(n).unwrap();

        // Dump the expression as a dot file
        expr.as_dot("/tmp/dot.dot");
    }

    #[test]
    fn it_works() {
        let mut expr = Expression::default();

        let xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ];

        let desired_targets = [1.0, -1.0, -1.0, 1.0];

        // Create the MLP for this example
        let mlp  = expr.new_mlp(xs[0].len(), &[4, 4, 1]);

        // Add the output prediction nodes to the expression
        let mut y_predictions = Vec::new();

        // 
        for x in xs {
            let out = expr.activate_mlp(&mlp, &x)[0];
            y_predictions.push(out);
        }

        // Add a loss node to the graph
        let mut loss = expr.new_value("loss", 0.0);
        for (target, prediction) in desired_targets.iter().zip(y_predictions.iter()) {
            let target = expr.new_value("_", *target);

            let sub = expr.sub("losssub", *prediction, target);
            let two = expr.new_value("two", 2.0);
            let pow = expr.pow("losspow", sub, two);

            loss = expr.add("loss", pow, loss);
        }

        println!("--- PREDICTIONS BEFORE ---");
        for (y_pred, desired) in y_predictions.iter().zip(desired_targets.iter()) {
            println!("{:11.8?} | wanted {:11.8?}", expr.get_value(&y_pred), desired);
        }

        #[derive(Debug)]
        enum Step {
            GetLoss,
            ZeroGrad,
            BackProp,
            Nudge,
            COUNT,
        }

        let mut cycles = [0; Step::COUNT as usize];

        macro_rules! time {
            ($step:ident, $work:expr) => {{
                let start = unsafe { std::arch::x86_64::_rdtsc() };

                let res = $work;

                cycles[Step::$step as usize] += unsafe { std::arch::x86_64::_rdtsc() } - start;

                res
            }}
        }

        // Optimize the expression, minimizing the loss
        let mut curr_loss = 10.0;
        let mut iters = 0;
        let init_start = unsafe { std::arch::x86_64::_rdtsc() };

        while curr_loss >= 0.001 { 
            curr_loss = time!(GetLoss, expr.get_value(&loss));

            // Naive nudge
            let nudge = if curr_loss <= 0.01 {
                -0.0001
            } else if curr_loss <= 0.1 {
                -0.001
            } else {
                -0.01
            };

            // Zero gradients
            time!(ZeroGrad, expr.zero_grad());

            // Back propagate
            time!(BackProp, expr.back_prop(loss).unwrap());

            // Nudge the expression by the newly found gradients
            time!(Nudge, expr.nudge_by(nudge));

            iters += 1;
        }

        let total_cycles = unsafe { std::arch::x86_64::_rdtsc() } - init_start;

        println!("--- Basic statistics of the optimization process --- ");

        println!("Cycles per optimization iter: {:6.2?}", total_cycles as f64 / iters as f64);
        macro_rules! stats {
            ($stat:ident) => {
                println!("    {:8} | {:6.4}%", 
                    format!("{:?}", Step::$stat),
                    cycles[Step::$stat as usize] as f64 / total_cycles as f64);
            }
        }
        stats!(GetLoss);
        stats!(ZeroGrad);
        stats!(BackProp);
        stats!(Nudge);
        
        println!("--- PREDICTIONS AFTER ---");
        for (y_pred, desired) in y_predictions.iter().zip(desired_targets.iter()) {
            println!("{:11.8?} | wanted {:11.8?}", expr.get_value(&y_pred), desired);
        }

        println!(".dot file written to /tmp/dot.dot");
        println!("dot -Tsvg /tmp/dot.dot > /tmp/dot.svg");
        expr.as_dot("/tmp/dot2.dot");
    }
}
