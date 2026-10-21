// A section and a key without doc comments.
use serde::Deserialize;
use syntropy_config::ConfigSection;

#[derive(Deserialize, ConfigSection)]
#[config(section = "probe", scope = Node, owner = "RFC-0010")]
struct NoSectionDoc {
    /// Documented.
    #[config(reload = Node, default = "7s")]
    model_wait: syntropy_config::Duration,
}

/// Documented.
#[derive(Deserialize, ConfigSection)]
#[config(section = "sched", scope = Node, owner = "RFC-0012")]
struct NoKeyDoc {
    #[config(reload = Hot, default = 4)]
    queue: u32,
}

fn main() {}
