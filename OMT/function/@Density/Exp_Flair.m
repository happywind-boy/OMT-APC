function DImg = Exp_Flair(Img, r, Mask)
    Flair = Img(:, :, :, 1);
    Flair = Channelwise_Z_Score(Flair);

    if ~isempty(Mask)
        Flair = Flair .* Mask;
    end

    DImg = exp(r * Flair);
    DImg = DImg / max(DImg, [], 'all');
end