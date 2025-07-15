use anyhow::Result;
use candle::{safetensors, Device};
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    let tensors = safetensors::load(&args.model, &device)?;
    for key in tensors.keys() {
        println!("{}", key);
    }
    Ok(())
}
