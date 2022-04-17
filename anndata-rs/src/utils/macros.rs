macro_rules! proc_arr_data {
    ($dtype:expr, $reader:expr, $fun:ident) => {
        match $dtype {
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U1) => {
                let mat: ArrayD<i8> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U2) => {
                let mat: ArrayD<i16> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U4) => {
                let mat: ArrayD<i32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U8) => {
                let mat: ArrayD<i64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U1) => {
                let mat: ArrayD<u8> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U2) => {
                let mat: ArrayD<u16> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U4) => {
                let mat: ArrayD<u32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U8) => {
                let mat: ArrayD<u64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U4) => {
                let mat: ArrayD<f32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
                let mat: ArrayD<f64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Boolean => {
                let mat: ArrayD<bool> = $reader;
                $fun!(mat)
            },
            other => panic!("type {} is not supported", other),
        }
    }
}

macro_rules! proc_csr_data {
    ($dtype:expr, $reader:expr) => {
        match $dtype {
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U1) => {
                let mat: CsrMatrix<i8> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U2) => {
                let mat: CsrMatrix<i16> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U4) => {
                let mat: CsrMatrix<i32> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U8) => {
                let mat: CsrMatrix<i64> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U1) => {
                let mat: CsrMatrix<u8> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U2) => {
                let mat: CsrMatrix<u16> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U4) => {
                let mat: CsrMatrix<u32> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U8) => {
                let mat: CsrMatrix<u64> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U4) => {
                let mat: CsrMatrix<f32> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
                let mat: CsrMatrix<f64> = $reader;
                Ok(Box::new(mat))
            },
            hdf5::types::TypeDescriptor::Boolean => {
                let mat: CsrMatrix<bool> = $reader;
                Ok(Box::new(mat))
            },
            other => panic!("type {} is not supported", other),
        }
    }
}

pub(crate) use proc_arr_data;
pub(crate) use proc_csr_data;