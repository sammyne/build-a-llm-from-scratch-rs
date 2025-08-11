fn main() {
    let text = "Hello, world. This, is a test.";

    let result: Vec<_> = text.split_ascii_whitespace().collect();
    println!("{result:?}");
}
