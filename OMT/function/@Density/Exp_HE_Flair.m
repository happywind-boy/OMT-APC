function DImg = Exp_HE_Flair(Img, r, Mask)
    Flair = Img(:,:,:,1);
    Flair = Channelwise_Z_Score(Flair);
    HR_Flair = Channelwise_HE(Flair);
    
    if ~isempty(Mask)
        HR_Flair = HR_Flair .* Mask;
    end
    
    DImg = exp(r * HR_Flair);
    DImg = DImg / max(DImg,[],'all');
end