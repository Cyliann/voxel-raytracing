use shaders::run;

fn main() {
    pollster::block_on(run());
}
