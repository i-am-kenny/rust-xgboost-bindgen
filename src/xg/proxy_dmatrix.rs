use std::ops::Deref;

use super::ArrayInterface;

pub struct ProxyDMatrix<'a, T> {
    pub(crate) inner: Data<'a, T>,
}

pub(crate) enum Data<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}
impl<T> Deref for Data<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match &self {
            Data::Owned(a) => a,
            Data::Borrowed(a) => a,
        }
    }
}

pub trait XGCompatible {
    fn hint(&self) -> XGMatrixType;
}

pub enum XGMatrixType {
    CudaDense(ArrayInterface),
}

impl<T: XGCompatible> ProxyDMatrix<'_, T> {
    pub fn owned(value: T) -> Self {
        Self {
            inner: Data::Owned(value),
        }
    }
}

impl<'a, T: XGCompatible> ProxyDMatrix<'a, T> {
    pub fn borrowed(value: &'a T) -> Self {
        Self {
            inner: Data::Borrowed(value),
        }
    }
}

/*
pub trait TypeStr {
    fn type_str() -> &'static str;
}

impl TypeStr for f32 {
    fn type_str() -> &'static str {
        "<f4"
    }
}

impl<A: TypeStr, D: ndarray::Dimension> XGCompatible for ndarray::Array<A, D> {
    fn hint(&self) -> XGMatrixType {
        let shape: Vec<_> = self.axes().map(|ax| ax.len as u64).collect();

        let dim = self.ndim();
        // let strides: Vec<_> = self
        //     .strides()
        //     .iter()
        //     .map(|s| (*s as usize) * dim * size_of::<A>())
        //     .collect();
        // let mut strides: Vec<_> = self.axes().map(|ax| ax.stride as usize).collect();
        // let mut strides: Vec<_> = shape
        //     .iter()
        //     .map(|ax| (*ax as usize) * size_of::<A>())
        //     .collect();
        //
        //
        // println!("axes: {shape:?}, strides: {strides:?}");
        // println!("size:{}, {strides:?}", size_of::<A>());
        // let strides = vec![4, 168348];

        // strides.reverse();

        let pointer = self.as_ptr() as usize;
        let interface = ArrayInterface {
            shape,
            typestr: A::type_str().into(),
            version: 3,
            data: Some(ArrayReference {
                pointer,
                read_only: true,
            }),
            descr: vec![vec!["", "<f4"]],
            // strides: Some(strides),
            strides: None,
        };
        XGMatrixType::Dense(interface)
    }

    // fn array_interface(&self) -> ArrayInterface {
    //     todo!()
    // }
}
*/
