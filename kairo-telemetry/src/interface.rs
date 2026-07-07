use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait Telemetry: Send + Sync {
    fn counter(&self, name: &'static str, value: u64);
    fn gauge(&self, name: &'static str, value: f64);
    fn histogram(&self, name: &'static str, value: f64);
}

pub struct TestTelemetry {
    counters: Arc<Mutex<HashMap<String, u64>>>,
    gauges: Arc<Mutex<HashMap<String, f64>>>,
    histograms: Arc<Mutex<HashMap<String, Vec<f64>>>>,
}

impl TestTelemetry {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
            gauges: Arc::new(Mutex::new(HashMap::new())),
            histograms: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.lock().unwrap().get(name).copied()
    }

    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.lock().unwrap().get(name).copied()
    }

    pub fn get_histogram(&self, name: &str) -> Option<Vec<f64>> {
        self.histograms.lock().unwrap().get(name).cloned()
    }
}

impl Default for TestTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry for TestTelemetry {
    fn counter(&self, name: &'static str, value: u64) {
        let mut counters = self.counters.lock().unwrap();
        *counters.entry(name.to_string()).or_insert(0) += value;
    }

    fn gauge(&self, name: &'static str, value: f64) {
        self.gauges.lock().unwrap().insert(name.to_string(), value);
    }

    fn histogram(&self, name: &'static str, value: f64) {
        self.histograms
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_default()
            .push(value);
    }
}

pub struct NoOpTelemetry;

impl Telemetry for NoOpTelemetry {
    fn counter(&self, _name: &'static str, _value: u64) {}
    fn gauge(&self, _name: &'static str, _value: f64) {}
    fn histogram(&self, _name: &'static str, _value: f64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_records_counter() {
        let telemetry = TestTelemetry::new();
        telemetry.counter("requests", 1);
        assert_eq!(telemetry.get_counter("requests"), Some(1));
    }

    #[test]
    fn test_telemetry_records_gauge_and_histogram() {
        let telemetry = TestTelemetry::new();
        telemetry.gauge("temperature", 22.5);
        telemetry.histogram("latency", 120.0);
        assert_eq!(telemetry.get_gauge("temperature"), Some(22.5));
        assert_eq!(telemetry.get_histogram("latency"), Some(vec![120.0]));
    }
}
