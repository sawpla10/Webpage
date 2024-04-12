import * as tf from '@tensorflow/tfjs';

// 이 함수는 TensorFlow 모델을 로드합니다
export const loadModel = async () => {
  const model = await tf.loadLayersModel('/path/to/model.json');
  return model;
};


// React 컴포넌트에서 사용하는 예시
import React, { useEffect } from 'react';
import { loadModel } from './useTensorFlow';

function App() {
  useEffect(() => {
    const fetchModel = async () => {
      const model = await loadModel();
      // 모델을 사용한 추후 로직...
    };

    fetchModel();
  }, []);

  return <div>스터디 추천 </div>;
}

export default App;
