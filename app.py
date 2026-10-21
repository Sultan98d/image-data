import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title='Handwritten Digits Explorer', page_icon='🔢', layout='wide')

@st.cache_data
def get_data():
    d = load_digits()
    return d.images, d.target

@st.cache_resource
def get_model():
    imgs, y = get_data()
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=2000, solver='lbfgs')
    model.fit(imgs[idx_train].reshape(-1,64)/16.0, y[idx_train])
    return model, idx_train, idx_test

images, labels = get_data()
model, train_idx, test_idx = get_model()

st.title('Handwritten Digits Explorer')
st.caption('DATS 6401 | Week 8 — Image Data Visualization')
st.write('Explore patterns in handwritten digits using montages, pixel-intensity visualizations, model saliency, and a controlled stamp experiment.')

section = st.sidebar.radio('Explore', ['Overview & Montage', 'Image Structure', 'Class Comparison', 'Saliency & Model', 'Stamp Experiment'])
st.sidebar.caption('Dataset: scikit-learn handwritten digits (8×8 grayscale)')

if section == 'Overview & Montage':
    st.header('Dataset Overview')
    a,b,c = st.columns(3)
    a.metric('Images', len(images)); b.metric('Image dimensions', '8 × 8'); c.metric('Classes', '10 (0–9)')
    st.write('Each image contains 64 grayscale intensity values from 0 (dark) to 16 (bright).')
    counts = np.bincount(labels, minlength=10)
    fig, ax = plt.subplots(figsize=(9,3))
    ax.bar(range(10), counts)
    ax.set(xlabel='Digit class', ylabel='Number of images', title='Class balance')
    ax.set_xticks(range(10)); st.pyplot(fig); plt.close(fig)
    st.write(f'Class counts range from {counts.min()} to {counts.max()} images; the dataset is approximately balanced.')
    st.header('Image Montage')
    digit = st.selectbox('Select digit class', ['All digits'] + list(range(10)))
    per_class = st.slider('Images per class (for all digits)', 2, 8, 5)
    if digit == 'All digits':
        selected = [np.flatnonzero(labels == k)[:per_class] for k in range(10)]
        fig, axes = plt.subplots(10, per_class, figsize=(per_class*1.35, 12))
        for k in range(10):
            for j, ix in enumerate(selected[k]):
                axes[k,j].imshow(images[ix], cmap='gray_r', vmin=0, vmax=16)
                axes[k,j].axis('off')
                if j == 0: axes[k,j].set_ylabel(str(k), rotation=0, labelpad=12)
        fig.suptitle('Rows represent digit classes (0–9)', fontsize=13)
    else:
        chosen = np.flatnonzero(labels == digit)[:24]
        fig, axes = plt.subplots(4,6, figsize=(10,7))
        for ax, ix in zip(axes.flat, chosen):
            ax.imshow(images[ix], cmap='gray_r', vmin=0, vmax=16)
            ax.set_title(f'#{ix}', fontsize=9); ax.axis('off')
        fig.suptitle(f'Variations in handwritten digit {digit}')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.info('Interpretation: Samples of the same digit vary in stroke thickness, position, and handwriting style. Comparing rows makes within-class variation and between-class shape differences visible.')

elif section == 'Image Structure':
    st.header('Image Structure & Pixel Intensity')
    ix = st.slider('Choose image index', 0, len(images)-1, 0)
    img = images[ix]
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(img, cmap='gray_r', vmin=0, vmax=16, interpolation='nearest')
        ax.set_title(f'Image #{ix}: digit {labels[ix]}'); ax.axis('off')
        st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(img, cmap='viridis', vmin=0, vmax=16, interpolation='nearest')
        ax.set_title('Pixel intensity heatmap')
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        fig.colorbar(im, ax=ax, label='Intensity (0–16)')
        st.pyplot(fig); plt.close(fig)
    st.subheader('Pixel Intensity Distribution')
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(img.ravel(), bins=np.arange(-0.5,17.5,1), edgecolor='black')
    ax.set(xlabel='Pixel intensity', ylabel='Number of pixels', title='Distribution across 64 pixels')
    ax.set_xticks(range(17)); st.pyplot(fig); plt.close(fig)
    st.write(f'**Mean intensity:** {img.mean():.2f} | **Nonzero pixels:** {np.count_nonzero(img)}/64 | **Maximum intensity:** {img.max():.0f}')
    st.info('Interpretation: Most background pixels have intensity 0. Higher values trace the handwritten strokes. The heatmap preserves pixel locations, while the histogram summarizes intensities but not their positions.')

elif section == 'Class Comparison':
    st.header('Compare Two Digit Classes')
    x,y = st.columns(2)
    with x: first = st.selectbox('First digit', range(10), index=3)
    with y: second = st.selectbox('Second digit', range(10), index=8)
    avg1 = images[labels == first].mean(axis=0)
    avg2 = images[labels == second].mean(axis=0)
    fig, axes = plt.subplots(1,3, figsize=(11,3.5))
    for ax, arr, title in zip(axes, [avg1,avg2,avg1-avg2], [f'Mean image: {first}', f'Mean image: {second}', 'Difference (first − second)']):
        if 'Difference' in title:
            m = ax.imshow(arr, cmap='coolwarm', vmin=-16, vmax=16)
        else: m = ax.imshow(arr, cmap='gray_r', vmin=0, vmax=16)
        ax.set_title(title); ax.axis('off'); fig.colorbar(m, ax=ax, fraction=.046)
    st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,3.5))
    bins=np.arange(-0.5,17.5,1)
    ax.hist(images[labels==first].ravel(), bins=bins, density=True, alpha=.55, label=f'Digit {first}')
    ax.hist(images[labels==second].ravel(), bins=bins, density=True, alpha=.55, label=f'Digit {second}')
    ax.set(xlabel='Pixel intensity', ylabel='Proportion (density)', title='Class pixel-intensity distributions')
    ax.legend(); st.pyplot(fig); plt.close(fig)
    st.info('Interpretation: Mean images show where strokes tend to appear in each class. The difference heatmap identifies pixels with different average intensities. Histograms show brightness frequencies, but cannot show spatial structure.')

elif section == 'Saliency & Model':
    st.header('Classification & Occlusion Saliency')
    predictions = model.predict(images[test_idx].reshape(-1,64)/16.0)
    accuracy = accuracy_score(labels[test_idx], predictions)
    st.metric('Held-out test accuracy', f'{accuracy:.1%}')
    st.caption('Stratified 75/25 train/test split (random_state=42). Test images were not used for fitting.')
    pos = st.slider('Select a held-out test image', 0, len(test_idx)-1, 0)
    ix = int(test_idx[pos]); img = images[ix]
    original_probs = model.predict_proba(img.reshape(1,-1)/16.0)[0]
    predicted = int(model.classes_[np.argmax(original_probs)])
    class_col = int(np.flatnonzero(model.classes_ == predicted)[0])
    original_conf = float(original_probs[class_col])
    drops = np.zeros((8,8))
    for r in range(8):
        for c in range(8):
            altered = img.copy(); altered[r,c] = 0
            new_prob = model.predict_proba(altered.reshape(1,-1)/16.0)[0,class_col]
            drops[r,c] = original_conf-new_prob
    a,b = st.columns(2)
    with a:
        fig,ax=plt.subplots(figsize=(5,4))
        ax.imshow(img,cmap='gray_r',vmin=0,vmax=16); ax.set_title(f'True digit: {labels[ix]} | Predicted: {predicted}'); ax.axis('off')
        st.pyplot(fig); plt.close(fig)
    with b:
        fig,ax=plt.subplots(figsize=(5,4))
        vmax=max(float(np.max(np.abs(drops))),1e-6)
        hm=ax.imshow(drops,cmap='coolwarm',vmin=-vmax,vmax=vmax)
        ax.set_title('Occlusion saliency: change in predicted-class probability')
        fig.colorbar(hm,ax=ax,label='Probability drop after masking pixel')
        st.pyplot(fig); plt.close(fig)
    strongest=np.unravel_index(np.argmax(drops),drops.shape)
    st.write(f'**Predicted-class probability:** {original_conf:.1%}. Largest positive drop: {drops[strongest]:.3f} at row {strongest[0]}, column {strongest[1]} (zero-indexed).')
    st.info('Interpretation: Positive (red) values indicate pixels whose removal lowers confidence in the original predicted class; negative (blue) values indicate removal increases confidence. This is a local sensitivity measure, not proof that the model reasons like a human. A single pixel is replaced by background intensity 0, and the prediction is recomputed.')
    st.subheader('Test-set confusion matrix')
    cm=confusion_matrix(labels[test_idx],predictions,labels=range(10))
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(cm,cmap='Blues'); ax.set(xlabel='Predicted digit',ylabel='True digit')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    fig.colorbar(im,ax=ax); st.pyplot(fig); plt.close(fig)

else:
    st.header('Optional Stamp Experiment: Shortcut Learning')
    st.write('We add an artificial bright pixel to digit 3 in the training data, then compare how the model behaves when the stamp is present or absent.')
    @st.cache_resource
    def train_stamped():
        stamped=images.copy()
        mask=(labels[train_idx]==3)
        stamped[train_idx[mask],0,0]=16
        trained=LogisticRegression(max_iter=2000,solver='lbfgs')
        trained.fit(stamped[train_idx].reshape(-1,64)/16.0,labels[train_idx])
        return trained
    stamped_model=train_stamped()
    clean_test=images[test_idx].reshape(-1,64)/16.0
    stamped_test=images[test_idx].copy()
    stamped_test[labels[test_idx]==3,0,0]=16
    stamped_test=stamped_test.reshape(-1,64)/16.0
    clean_preds=stamped_model.predict(clean_test)
    stamp_preds=stamped_model.predict(stamped_test)
    mask3=labels[test_idx]==3
    recall_clean=float(np.mean(clean_preds[mask3]==3))
    recall_stamp=float(np.mean(stamp_preds[mask3]==3))
    c1,c2=st.columns(2)
    c1.metric('Digit 3 recall — stamp absent',f'{recall_clean:.1%}')
    c2.metric('Digit 3 recall — stamp present',f'{recall_stamp:.1%}')
    st.write(f'Overall test accuracy (clean): {accuracy_score(labels[test_idx],clean_preds):.1%}; with stamp on test digit 3: {accuracy_score(labels[test_idx],stamp_preds):.1%}.')
    ix=int(test_idx[np.flatnonzero(mask3)[0]])
    example=images[ix].copy(); marked=example.copy(); marked[0,0]=16
    fig,axes=plt.subplots(1,2,figsize=(7,3))
    for ax,img,title in zip(axes,[example,marked],['Clean digit 3','Digit 3 with stamp']):
        ax.imshow(img,cmap='gray_r',vmin=0,vmax=16); ax.set_title(title); ax.axis('off')
    st.pyplot(fig); plt.close(fig)
    if recall_stamp>recall_clean:
        st.warning('Data-quality finding: Recognition of digit 3 improves when the artificial stamp is present. The model appears to rely partly on a non-semantic shortcut instead of digit shape alone.')
    elif recall_stamp<recall_clean:
        st.info('Finding: The stamp reduces digit-3 recall in this run. Artificial artifacts can alter behavior, but this experiment does not show the expected shortcut improvement.')
    else:
        st.info('Finding: Digit-3 recall is unchanged in this run. This does not rule out smaller changes in prediction confidence or other forms of artifact sensitivity.')
    st.caption('Controlled demonstration: the stamp is added only to digit-3 training images; held-out test images are compared with and without the same stamp. Results are computed from the model rather than hard-coded.')

st.divider()
st.caption('Data source: scikit-learn load_digits | All analyses use the same 1,797-image dataset.')
