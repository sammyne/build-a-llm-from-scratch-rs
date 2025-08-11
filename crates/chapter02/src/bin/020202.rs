use std::fs;

/// Listing 2.1 Reading in a short story as text sample into Python
fn main() {
    let raw_text = fs::read_to_string("static/the-verdict.txt").expect("Failed to read file");
    println!("Total number of character: {}", raw_text.chars().count());
    println!("{}", raw_text.split_at(99).0)
}
