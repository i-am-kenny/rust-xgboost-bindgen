use serde::{ser::SerializeSeq, Serialize};

/// https://numpy.org/doc/2.1/reference/arrays.interface.html
#[derive(Serialize, Clone, Debug)]
pub struct ArrayInterface {
    pub shape: Vec<u64>,

    // typestr: ArrayType
    pub typestr: String,

    pub version: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<ArrayReference>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub strides: Option<Vec<usize>>,

    pub descr: Vec<Vec<&'static str>>
}

#[derive(Debug, Clone)]
pub struct ArrayReference {
    pub pointer: usize,

    pub read_only: bool,
}

impl Serialize for ArrayReference {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut array = serializer.serialize_seq(Some(2))?;
        array.serialize_element(&self.pointer)?;
        array.serialize_element(&self.read_only)?;

        array.end()
    }
}
